import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score

from src.models.optim.CustomLosses import DeepSVDDLoss, DeepSVDDLossSubspace, MaskedMSELoss
from src.utils.utils import print_progessbar, get_best_threshold

class DeepSVDD_Joint_trainer:
    """
    Trainer for the DeepSVDD together with the autoencoder (joint training).
    """
    def __init__(self, space_repr, nu, R, lr=0.0001, n_epoch=150, n_epoch_pretrain=10, n_epoch_warm_up=10,
                 lr_milestone=(), batch_size=64,
                 weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False,
                 criterion_weight=(0.5,0.5), soft_boundary=False, use_subspace=False):
        """
        Build a Joint DeepSVDD and AutoEncoder trainer.
        ----------
        INPUT
            |---- space_repr (torch.Tensor) the hypersphere center or the projection
            |           matrix on the training normal samples.
            |---- nu (float) the DeepSVDD parameter the a priory fraction of outlier
            |           in the train set.
            |---- R (float) the radius for the potential soft-boundary.
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
            |---- n_epoch_warm_up
            |---- n_epoch_pretrain
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
            |---- criterion_weight (tuple (weight reconstruction, weight SVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           Deep SVDD).
            |---- soft_boundary (bool) whether to use soft boundary.
            |---- use_subspace (bool) whether to use the subspace projecttion as
            |           a distance metric for the SVDD loss computation.
        OUTPUT
            |---- None
        """
        # learning parameters
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_epoch_pretrain = n_epoch_pretrain
        self.n_epoch_warm_up = n_epoch_warm_up
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.print_batch_progress = print_batch_progress
        self.criterion_weight = criterion_weight
        self.use_subspace = use_subspace
        self.soft_boundary = soft_boundary

        # define the loss to use on the SVDD embedding
        self.SVDDLoss = DeepSVDDLossSubspace if use_subspace else DeepSVDDLoss

        self.scale_rec = 1.0
        self.scale_em = 1.0

        # DeepSVDD parameters
        self.space_repr = torch.tensor(space_repr, device=self.device) if space_repr is not None else None
        self.nu = nu
        self.R = torch.tensor(R, device=self.device)

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
        Pretrain the AE for the joint DeepSVDD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSVDD to train. The network should be an
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
        Train the joint DeepSVDD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSVDD to train. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- net (nn.Module) The trained joint DeepSVDD.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)

        # put net to device
        net = net.to(self.device)
        net.return_svdd_embed = True # enable the network to provide the SVDD embdeding

        # initialize hypersphere center or subspace projection matrix
        if self.space_repr is None:
            if self.use_subspace:
                logger.info('>>> Initializing the subspace projection matrix.')
                self.space_repr = self.initialize_projection_matrix(train_loader, net)
                logger.info('>>> Projection matrix succesfully initialized.')
            else:
                logger.info('>>> Initializing the hypersphere center.')
                self.space_repr = self.initialize_hypersphere_center(train_loader, net)
                logger.info('>>> Center succesfully initialized.')

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss()
        criterion_ad = self.SVDDLoss(self.space_repr, self.nu, eps=self.eps, soft_boundary=self.soft_boundary)

        # compute the scale weight so that the rec and svdd losses are scalled and comparable
        self.initialize_loss_scale_weight(train_loader, net, criterion_rec, criterion_ad)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('>>> Start Training the Joint DeepSVDD and Autoencoder.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch_tot = train_loader.__len__()
        # set network in train mode
        net.train()
        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            n_batch = 0
            epoch_start_time = time.time()
            dist = []

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
                loss_ad = criterion_ad(embed, self.R)
                loss_ad = self.scale_em * self.criterion_weight[1] * loss_ad
                loss = loss_rec + loss_ad

                loss.backward()
                optimizer.step()

                # compute dist to update radius R
                if self.soft_boundary and (epoch + 1 > self.n_epoch_warm_up):
                    if self.use_subspace:
                        dist.append(torch.sum((embed - torch.matmul(self.space_repr, embed.transpose(0,1)).transpose(0,1))**2, dim=1).detach())
                    else:
                        dist.append(torch.sum((self.space_repr - embed)**2, dim=1).detach())


                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

            # update radius
            if self.soft_boundary and (epoch + 1 > self.n_epoch_warm_up):
                self.R.data = torch.tensor(self.get_radius(torch.cat(dist, dim=0)), device=self.device)

            # epoch statistic
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch:03} | Train Time: {epoch_train_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} |')

            # append the epoch loss to results list
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update the learning rate if the milestone is reached
            scheduler.step()
            if epoch + 1 in self.lr_milestone:
                logger.info(f'>>> LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

        # End training
        self.train_loss = epoch_loss_list
        self.train_time = time.time() - start_time
        logger.info(f'>>> Training of Joint DeepSVDD and AutoEncoder Time: {self.train_time:.3f} [s]')
        logger.info('>>> Finished Joint DeepSVDD and AutoEncoder Training.\n')

        return net

    def initialize_loss_scale_weight(self, loader, net, criterion_rec, criterion_ad):
        """
        Perform one forward pass to compute the reconstruction and embdeding loss
        scalling factors to get a loss in the magnitude of 1.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DeepSVDD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- criterion_rec (nn.Module) the reconstruction loss criterion.
            |---- criterion_ad (nn.Module) the SVDD embdeding loss criterion.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()
        logger.info('>>> Initializing the loss scale factors.')
        sumloss_rec = 0.0
        sumloss_ad = 0.0
        n_batch = 0
        net.train()
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
                loss_ad = criterion_ad(embed, self.R)
                sumloss_ad += loss_ad.item()

                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, loader.__len__(), Name='\t\tBatch', Size=20)

            # initialize the scalling weight of the reconstruction loss so that it's 1 at epoch 1
            self.scale_rec = 1 / (sumloss_rec / n_batch)
            self.scale_em = 1 / (sumloss_ad / n_batch)
            logger.info(f'>>> reconstruction loss scale factor initialized to {self.scale_rec:.6f}')
            logger.info(f'>>> SVDD embdeding loss scale factor initialized to {self.scale_em:.6f}')

    def get_radius(self, dist):
        """
        Optimally solve for radius R via the (1-nu)-quantile of distances.
        """
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)

    def validate(self, dataset, net):
        """
        Validate the joint DeepSVDD network on the provided dataset and find the
        best threshold on the score to maximize the f1-score.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSVDD to validate. The network should be
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
        criterion_ad = self.SVDDLoss(self.space_repr, self.nu, eps=self.eps, soft_boundary=self.soft_boundary)

        # Testing
        logger.info('>>> Start Validating of the joint DeepSVDD and AutoEncoder.')
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
                loss_ad = criterion_ad(embed, self.R)
                # compute anomaly scores
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch
                #rec_score = torch.sum(loss_rec, dim=tuple(range(1, rec.dim()))) / (torch.sum(mask, dim=tuple(range(1, rec.dim()))) + 1) # mean reconstruction MSE on the mask per batch
                if self.use_subspace:
                    dist = torch.sum((embed - torch.matmul(self.space_repr, embed.transpose(0,1)).transpose(0,1)) ** 2, dim=1) # score is the distance (large distances highlight anomalies)
                else:
                    dist = torch.sum((embed - self.space_repr) ** 2, dim=1) # score is the distance (large distances highlight anomalies)

                if self.soft_boundary:
                    ad_score = dist - self.R ** 2
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
        logger.info(f'>>> Validation DeepSVDD AUC: {self.valid_auc_ad:.3%}')
        logger.info(f'>>> Best Threshold for the DeepSVDD score maximizing F1-score: {self.scores_threhold_ad:.3f}')
        logger.info(f'>>> Best F1-score on DeepSVDD score: {self.valid_f1_ad:.3%}')
        logger.info('>>> Finished validating the Joint DeepSVDD and AutoEncoder.\n')

    def test(self, dataset, net):
        """
        Test the joint DeepSVDD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSVDD to test. The network should be an
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
        criterion_ad = self.SVDDLoss(self.space_repr, self.nu, eps=self.eps, soft_boundary=self.soft_boundary)

        # Testing
        logger.info('>>> Start Testing the joint DeepSVDD and AutoEncoder.')
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
                loss_ad = criterion_ad(embed, self.R)
                # compute anomaly scores
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch
                if self.use_subspace:
                    dist = torch.sum((embed - torch.matmul(self.space_repr, embed.transpose(0,1)).transpose(0,1)) ** 2, dim=1) # score is the distance (large distances highlight anomalies)
                else:
                    dist = torch.sum((embed - self.space_repr) ** 2, dim=1) # score is the distance (large distances highlight anomalies)

                if self.soft_boundary:
                    ad_score = dist - self.R ** 2
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
        logger.info(f'>>> Test AD AUC: {self.test_auc_ad:.3%}')
        logger.info(f'>>> Test F1-score on DeepSVDD score: {self.test_f1_ad:.3%}')
        logger.info('>>> Finished Testing the Joint DeepSVDD and AutoEncoder.\n')

    def initialize_hypersphere_center(self, loader, net, eps=0.1):
        """
        Initialize the hypersphere center as the mean output of the network over
        one forward pass.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DeepSVDD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- eps (float) the epsilon representing the minimum value of the
            |           component of the center.
        OUTPUT
            |---- c (torch.Tensor) the initialized center.
        """
        n_sample = 0
        # get embdedding dimension with one forward pass of one batch
        with torch.no_grad():
            sample = next(iter(loader))[0].float()
            svdd_embedding_dim = net(sample.to(self.device))[1].shape[1]
        # initialize center
        c = torch.zeros(svdd_embedding_dim, device=self.device)

        # get the output of all samples and accumulate them
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, _, mask, _, _ = data
                input, mask = input.to(self.device).float(), mask.to(self.device)
                # mask input
                input = input * mask

                _, embed = net(input)
                n_sample += embed.shape[0]
                c += torch.sum(embed, dim=0)

                if self.print_batch_progress:
                    print_progessbar(b, loader.__len__(), Name='\t\tBatch', Size=20)

        # take the mean of accumulated c
        c /= n_sample
        # check if c_i are epsilon too close to zero to avoid them to be trivialy matched to zero
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c > 0)] = eps

        return c

    def initialize_projection_matrix(self, loader, net):
        """
        Initialize the subspace projection matrix from 2000 normal samples. The
        resulting matrix project an embeded point onto the estimated subspace of
        normal samples.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DeepSVDD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- eps (float) the epsilon representing the minimum value of the
            |           component of the center.
        OUTPUT
            |---- c (torch.Tensor) the initialized center.
        """
        N = 10000 #loader.dataset.__len__() # number of sample to use

        # Get S : matrix of sample embeding (M x N) with M the embeding dimension and N the number of samples
        S = []
        n_sample = 0
        net.eval()
        with torch.no_grad():
            for data in loader:
                input, _, mask, semi_label, _ = data
                input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)
                # mask input
                input = input * mask
                _, embed = net(input)
                embed = embed[semi_label != -1, :] # keep only embeding point of normal points
                S.append(embed)
                n_sample += embed.shape[0]

                if self.print_batch_progress:
                     print_progessbar(n_sample, N, Name='\t\tSamples', Size=20)

                if n_sample >= N:
                    break

        S = torch.cat(S, dim=0).transpose(0,1)
        S = S.to(self.device)

        # compute P = S(S'S + lI)^-1 S'
        inv = torch.inverse(torch.matmul(S.transpose(0,1), S) + 1e-3*torch.eye(S.shape[1], device=self.device))
        P = torch.matmul(S, torch.matmul(inv, S.transpose(0,1)))

        return P
