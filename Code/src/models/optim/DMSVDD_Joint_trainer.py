import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans

from src.models.optim.CustomLosses import DMSVDDLoss, MaskedMSELoss
from src.utils.utils import print_progessbar, get_best_threshold
from src.utils.kmeans_batch import Batch_KMeans

class DMSVDD_Joint_trainer:
    """

    """
    def __init__(self, c, nu, R, lr=0.0001, n_epoch=150, n_epoch_pretrain=10, n_epoch_warm_up=10,
                 lr_milestone=(), batch_size=64, n_sphere_init=100, weight_decay=1e-6,
                 device='cuda', n_jobs_dataloader=0, print_batch_progress=False,
                 criterion_weight=(0.5,0.5), soft_boundary=False):
        """

        """
        # learning parameters
        self.lr = lr
        self.lr_milestone = lr_milestone
        self.n_epoch = n_epoch
        self.n_epoch_pretrain = n_epoch_pretrain
        self.n_epoch_warm_up = n_epoch_warm_up
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.print_batch_progress = print_batch_progress
        self.criterion_weight = criterion_weight
        self.soft_boundary = soft_boundary

        self.n_sphere_init = n_sphere_init

        self.scale_rec = 1.0
        self.scale_em = 1.0

        # DMSVDD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.R = torch.zeros(self.n_sphere_init, device=self.device) if R is None else torch.tensor(R, device=self.device)

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.train_loss = None
        self.pretrain_time = None
        self.pretrain_loss = None

        self.valid_auc_rec = None
        self.valid_auc_ad = None
        self.valid_f1_rec = None
        self.valid_f1_ad = None
        self.valid_time = None
        self.valid_scores_rec = None
        self.valid_scores_ad = None

        self.test_auc_rec = None
        self.test_auc_ad = None
        self.test_f1_rec = None
        self.test_f1_ad = None
        self.test_time = None
        self.test_scores_rec = None
        self.test_scores_ad = None

        # threhold to define if anomalous
        self.scores_threhold_rec = None
        self.scores_threhold_ad = None

    def pretrain(self, dataset, net):
        """
        Pretrain the AE for the joint DMSVDD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervized labels.
            |---- net (nn.Module) The DMSVDD to train. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- net (nn.Module) The pretrained joint DeepSVDD.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net to device
        net = net.to(self.device)
        net.return_svdd_embed = False

        # define the two criterion for reconstruction
        criterion_rec = MaskedMSELoss()

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Start training
        logger.info('>>> Start Pretraining the Autoencoder.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch_tot = train_loader.__len__()

        # set network in train mode
        net.train()
        for epoch in range(self.n_epoch_pretrain):
            epoch_loss = 0.0
            n_batch = 0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                input, _, mask, semi_label, _ = data
                # put inputs to device
                input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)
                input.requires_grad = True

                # mask the input (keep only the object)
                input = input * mask

                # zeros the gradient
                optimizer.zero_grad()

                # Update network parameters by backpropagation
                rec, _ = net(input)
                # ignore reconstruction for known abnormal samples (no gradient update because loss = 0)
                rec = torch.where(semi_label.view(-1,1,1,1).expand(*input.shape) != -1, rec, input)
                loss = criterion_rec(rec, input, mask)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

            # epoch statistic
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch_pretrain:03} | Pretrain Time: {epoch_train_time:.3f} [s] '
                        f'| Pretrain Loss: {epoch_loss / n_batch:.6f} |')

            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

        # End training
        self.pretrain_loss = epoch_loss_list
        self.pretrain_time = time.time() - start_time
        logger.info(f'>>> Pretraining of AutoEncoder Time: {self.pretrain_time:.3f} [s]')
        logger.info('>>> Finished of AutoEncoder Pretraining.\n')

        return net

    def train(self, dataset, net):
        """
        Train the DMSVDD on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervized labels.
            |---- net (nn.Module) The DMSVDD to train. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- net (nn.Module) The pretrained joint DMSVDD.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)

        # put net to device
        net = net.to(self.device)
        net.return_svdd_embed = True # enable the network to provide the SVDD embdeding

        # initialize hypersphere center or subspace projection matrix
        if self.c is None:
            logger.info('>>> Initializing the hyperspheres centers.')
            self.initialize_centers(train_loader, net)
            logger.info(f'>>> {self.n_sphere_init} centers succesfully initialized.')

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss()
        criterion_ad = DMSVDDLoss(self.nu, eps=self.eps, soft_boundary=self.soft_boundary)

        # compute the scale weight so that the rec and svdd losses are scalled and comparable
        self.initialize_loss_scale_weight(train_loader, net, criterion_rec, criterion_ad)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('>>> Start Training the Joint DMSVDD and Autoencoder.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch_tot = train_loader.__len__()
        # set network in train mode
        net.train()
        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            n_batch = 0
            epoch_start_time = time.time()

            # update network
            for b, data in enumerate(train_loader):
                input, _, mask, semi_label, _ = data
                # put inputs to device
                input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)
                input.requires_grad = True

                # mask the input (keep only the object)
                input = input * mask

                # zeros the gradient
                optimizer.zero_grad()

                # Update network parameters by backpropagation on the two criterion
                rec, embed = net(input)
                # reconstruction loss
                # ignore reconstruction for known abnormal samples (no gradient update because loss = 0)
                rec = torch.where(semi_label.view(-1,1,1,1).expand(*input.shape) != -1, rec, input)
                loss_rec = criterion_rec(rec, input, mask)
                loss_rec = self.scale_rec * self.criterion_weight[0] * loss_rec
                # SVDD embedding loss
                loss_ad = criterion_ad(embed, self.c, self.R)
                loss_ad = self.scale_em * self.criterion_weight[1] * loss_ad
                loss = loss_rec + loss_ad

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tWeight-update Batch', Size=20)

            with torch.no_grad():
                # update radius R
                if epoch >= self.n_epoch_warm_up:
                    n_k = torch.zeros(self.c.shape[0], device=self.device)
                    dist = [[] for _ in range(self.c.shape[0])] # list of list for each center : N_center x N_k

                    for b, data in enumerate(train_loader):
                        # compute distance and belonging of sample
                        input, _, mask, semi_label, _ = data
                        input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)

                        # mask the input (keep only the object)
                        input = (input * mask)[semi_label != -1]
                        _, embed = net(input)
                        # get closest centers
                        min_dist, idx = torch.min(torch.norm(self.c.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=2), dim=1)
                        for i, d in zip(idx, min_dist):
                            n_k[i] += 1
                            dist[i].append(d)

                        if self.print_batch_progress:
                            print_progessbar(b, n_batch_tot, Name='\t\tRadius-update Batch', Size=20)

                    if self.soft_boundary:
                        # update R with (1-nu)th quantile
                        self.R = torch.where(n_k < self.nu * torch.max(n_k),
                                             torch.Tensor([0.0]).to(self.device),
                                             torch.Tensor([np.quantile(torch.stack(d, dim=0).clone().cpu().numpy(), 1 - self.nu) if len(d) > 0 else 0.0 for d in dist]).to(self.device))

                        # keep only centers and radius where R > 0
                        self.c = self.c[self.R > 0.0]
                        self.R = self.R[self.R > 0.0]
                    else:
                        # keep only centers that are not represented
                        self.c = self.c[n_k == 0] #self.c = self.c[n_k < self.nu * torch.max(n_k)]

            # epoch statistic
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train Time: {epoch_train_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} '
                        f'| N spheres: {self.c.shape[0]:03} |')

            # append the epoch loss to results list
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update the learning rate if the milestone is reached
            scheduler.step()
            if epoch + 1 in self.lr_milestone:
                logger.info(f'>>> LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

        # End training
        self.train_loss = epoch_loss_list
        self.train_time = time.time() - start_time
        logger.info(f'>>> Training of Joint DMSVDD and AutoEncoder Time: {self.train_time:.3f} [s]')
        logger.info('>>> Finished Joint DMSVDD and AutoEncoder Training.\n')

        return net

    def initialize_loss_scale_weight(self, loader, net, criterion_rec, criterion_ad):
        """
        Perform one forward pass to compute the reconstruction and embdeding loss
        scalling factors to get a loss in the magnitude of 1.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DMSVDD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- criterion_rec (nn.Module) the reconstruction loss criterion.
            |---- criterion_ad (nn.Module) the MSVDD embdeding loss criterion.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()
        logger.info('>>> Initializing the loss scale factors.')
        sumloss_rec = 0.0
        sumloss_ad = 0.0
        n_batch = 0
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, _, mask, semi_label, _ = data
                # put inputs to device
                input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)
                # mask input
                input = input * mask
                # forward
                rec, embed = net(input)
                # compute rec loss
                rec = torch.where(semi_label.view(-1,1,1,1).expand(*input.shape) != -1, rec, input) # ignore knwon abnormal samples
                loss_rec = criterion_rec(rec, input, mask)

                sumloss_rec += loss_rec.item()
                # compute ad loss
                loss_ad = criterion_ad(embed, self.c, self.R)
                sumloss_ad += loss_ad.item()

                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, loader.__len__(), Name='\t\tBatch', Size=20)

            # initialize the scalling weight of the reconstruction loss so that it's 1 at epoch 1
            self.scale_rec = 1 / (sumloss_rec / n_batch)
            self.scale_em = 1 / (sumloss_ad / n_batch)
            logger.info(f'>>> reconstruction loss scale factor initialized to {self.scale_rec:.6f}')
            logger.info(f'>>> MSVDD embdeding loss scale factor initialized to {self.scale_em:.6f}')

    def initialize_centers(self, loader, net, eps=0.1):
        """
        Initialize the multiple centers using the K-Means algorithm on the
        embedding of all the samples.
        ----------
        INPUT
            |---- loader (torch.utils.data.Dataloader) the loader of the data.
            |---- net (nn.Module) the DMSAD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- eps (float) minimal value for center coordinates, to avoid
            |           center too close to zero.
        OUTPUT
            |---- None
        """
        # K-means
        repr = []
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, _, mask, semi_label, _ = data
                input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)

                input = (input * mask)[semi_label != -1] # keep normal samples
                _, embed = net(input)
                repr.append(embed)

                if self.print_batch_progress:
                    print_progessbar(b, loader.__len__(), Name='\t\tBatch', Size=20)

            repr = torch.cat(repr, dim=0).cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_sphere_init, verbose=0).fit(repr)
        self.c = torch.tensor(kmeans.cluster_centers_).to(device=self.device)

        # get centers via batch kmeans algorithm
        # kmeans = Batch_KMeans(self.n_sphere_init, batch_size=self.batch_size,
        #                       n_jobs_dataloader=self.n_jobs_dataloader)
        # kmeans.fit(loader, net)
        # self.c = kmeans.c

        # check if c_i are epsilon too close to zero to avoid them to be trivialy matched to zero
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps

    def validate(self, dataset, net):
        """
        Validate the joint DMSVDD network on the provided dataset and find the
        best threshold on the score to maximize the f1-score.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DMSVDD to validate. The network should be
            |           an autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()

        # make test dataloader using image and mask
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.n_jobs_dataloader)

        # put net to device
        net = net.to(self.device)
        net.return_svdd_embed = True

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss(reduction='none')
        criterion_ad = DMSVDDLoss(self.nu, eps=self.eps, soft_boundary=self.soft_boundary)

        # Testing
        logger.info('>>> Start Validating of the joint DMSVDD and AutoEncoder.')
        epoch_loss = 0.0
        n_batch = 0
        n_batch_tot = valid_loader.__len__()
        start_time = time.time()
        idx_label_score_rec = []
        idx_label_score_ad = []
        # put network in evaluation mode
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(valid_loader):
                input, label, mask, semi_label, idx = data
                # put data to device
                input, label = input.to(self.device).float(), label.to(self.device)
                mask, semi_label = mask.to(self.device), semi_label.to(self.device)
                idx = idx.to(self.device)

                # mask the input
                input = input * mask

                # compute loss
                rec, embed = net(input)
                loss_rec = criterion_rec(rec, input, mask)
                loss_ad = criterion_ad(embed, self.c, self.R)
                # compute anomaly scores
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch

                dist, idx = torch.min(torch.sum((self.c.unsqueeze(0) - embed.unsqueeze(1))**2, dim=2), dim=1) # dist and idx by batch
                if self.soft_boundary:
                    ad_score = dist - torch.stack([self.R[i] ** 2 for i in idx], dim=0) #dist - self.R ** 2 --> negative = normal ; positive = abnormal
                else:
                    ad_score = dist

                # compute overall loss
                mean_loss_rec = torch.sum(loss_rec) / torch.sum(mask)
                loss = self.scale_rec * self.criterion_weight[0] * mean_loss_rec
                loss += self.scale_em * self.criterion_weight[1] * loss_ad

                # append scores and label
                idx_label_score_rec += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            rec_score.cpu().data.numpy().tolist()))
                idx_label_score_ad += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            ad_score.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

        self.valid_time = time.time() - start_time
        self.valid_scores_rec = idx_label_score_rec
        _, label, rec_score = zip(*idx_label_score_rec)
        label, rec_score = np.array(label), np.array(rec_score)
        self.valid_auc_rec = roc_auc_score(label, rec_score)
        self.scores_threhold_rec, self.valid_f1_rec = get_best_threshold(rec_score, label, metric=f1_score)

        self.valid_scores_ad = idx_label_score_ad
        _, label, ad_score = zip(*idx_label_score_ad)
        label, ad_score = np.array(label), np.array(ad_score)
        self.valid_auc_ad = roc_auc_score(label, ad_score)
        self.scores_threhold_ad, self.valid_f1_ad = get_best_threshold(ad_score, label, metric=f1_score)

        # add info to logger
        logger.info(f'>>> Validation Time: {self.valid_time:.3f} [s]')
        logger.info(f'>>> Validation Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Validation reconstruction AUC: {self.valid_auc_rec:.3%}')
        logger.info(f'>>> Best Threshold for the reconstruction score maximizing F1-score: {self.scores_threhold_rec:.3f}')
        logger.info(f'>>> Best F1-score on reconstruction score: {self.valid_f1_rec:.3%}')
        logger.info(f'>>> Validation DMSVDD AUC: {self.valid_auc_ad:.3%}')
        logger.info(f'>>> Best Threshold for the DMSVDD score maximizing F1-score: {self.scores_threhold_ad:.3f}')
        logger.info(f'>>> Best F1-score on DMSVDD score: {self.valid_f1_ad:.3%}')
        logger.info('>>> Finished validating the Joint DMSVDD and AutoEncoder.\n')

    def test(self, dataset, net):
        """
        Test the joint DMSVDD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DMSVDD to test. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()

        # make test dataloader using image and mask
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.n_jobs_dataloader)

        # put net to device
        net = net.to(self.device)
        net.return_svdd_embed = True

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss(reduction='none')
        criterion_ad = DMSVDDLoss(self.nu, eps=self.eps, soft_boundary=self.soft_boundary)

        # Testing
        logger.info('>>> Start Testing the joint DMSVDD and AutoEncoder.')
        epoch_loss = 0.0
        n_batch = 0
        n_batch_tot = test_loader.__len__()
        start_time = time.time()
        idx_label_score_rec = []
        idx_label_score_ad = []
        # put network in evaluation mode
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(test_loader):
                input, label, mask, semi_label, idx = data
                # put data to device
                input, label = input.to(self.device).float(), label.to(self.device)
                mask, semi_label = mask.to(self.device), semi_label.to(self.device)
                idx = idx.to(self.device)

                # mask the input
                input = input * mask

                # compute loss
                rec, embed = net(input)
                loss_rec = criterion_rec(rec, input, mask)
                loss_ad = criterion_ad(embed, self.c, self.R)
                # compute anomaly scores
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch

                dist, sphere_idx = torch.min(torch.sum((self.c.unsqueeze(0) - embed.unsqueeze(1))**2, dim=2), dim=1) # dist and idx by batch
                if self.soft_boundary:
                    ad_score = dist - torch.stack([self.R[i] ** 2 for i in sphere_idx], dim=0) #dist - self.R ** 2 --> negative = normal ; positive = abnormal
                else:
                    ad_score = dist

                # get overall loss
                mean_loss_rec = torch.sum(loss_rec) / torch.sum(mask)
                loss = self.scale_rec * self.criterion_weight[0] * mean_loss_rec
                loss += self.scale_em * self.criterion_weight[1] * loss_ad

                # append scores and label
                idx_label_score_rec += list(zip(idx.cpu().data.numpy().tolist(),
                                                label.cpu().data.numpy().tolist(),
                                                rec_score.cpu().data.numpy().tolist()))
                idx_label_score_ad += list(zip(idx.cpu().data.numpy().tolist(),
                                               label.cpu().data.numpy().tolist(),
                                               ad_score.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

        self.test_time = time.time() - start_time
        self.test_scores_rec = idx_label_score_rec
        _, label, rec_score = zip(*idx_label_score_rec)
        label, rec_score = np.array(label), np.array(rec_score)
        self.test_auc_rec = roc_auc_score(label, rec_score)
        self.test_f1_rec = f1_score(label, np.where(rec_score > self.scores_threhold_rec, 1, 0))

        self.test_scores_ad = idx_label_score_ad
        _, label, ad_score = zip(*idx_label_score_ad)
        label, ad_score = np.array(label), np.array(ad_score)
        self.test_auc_ad = roc_auc_score(label, ad_score)
        self.test_f1_ad = f1_score(label, np.where(ad_score > self.scores_threhold_ad, 1, 0))

        # add info to logger
        logger.info(f'>>> Test Time: {self.test_time:.3f} [s]')
        logger.info(f'>>> Test Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Test reconstruction AUC: {self.test_auc_rec:.3%}')
        logger.info(f'>>> Test F1-score on reconstruction score: {self.test_f1_rec:.3%}')
        logger.info(f'>>> Test DMSVDD AUC: {self.test_auc_ad:.3%}')
        logger.info(f'>>> Test F1-score on DMSVDD score: {self.test_f1_ad:.3%}')
        logger.info('>>> Finished Testing the Joint DMSVDD and AutoEncoder.\n')
