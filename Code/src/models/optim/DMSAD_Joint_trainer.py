import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans

from src.models.optim.CustomLosses import DMSADLoss, MaskedMSELoss
from src.utils.utils import print_progessbar, get_best_threshold

class DMSAD_Joint_trainer:
    """
    Define a trainer for the Deep multisphere SAD model which enables to train,
    validate and test the network.
    """
    def __init__(self, c, eta, n_sphere_init=100, n_epoch=150, n_epoch_pretrain=10,
                 lr=1e-4, weight_decay=1e-6, lr_milestone=(), criterion_weight=(0.5, 0.5),
                 batch_size=64, n_jobs_dataloader=0, device='cuda',
                 print_batch_progress=False):
        """
        Build a trainer for the DMSAD model.
        ----------
        INPUT
            |---- c (array like N_sphere x Embed dim) the centers of the hyperspheres.
            |           If None, the centers are initialized using Kmeans.
            |---- eta (float) the weight of semi-supervised labels in the loss.
            |---- n_sphere_init (int) the number of initial hypersphere.
            |---- n_epoch (int) the number of epoch.
            |---- n_epoch_pretrain (int) the number of epoch to perform with only
            |           the reconstruction loss.
            |---- lr (float) the learning rate.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- lr_milestone (tuple) the lr update steps (90% recudtion at each step).
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSAD).
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        # DMSAD parameters
        self.c = torch.tensor(c, device=device) if c else None
        self.eta = eta
        self.n_sphere_init = n_sphere_init
        self.alpha = 0.025 # minimum fraction of data per sphere

        # Learning parameters
        self.n_epoch = n_epoch
        self.n_epoch_pretrain = n_epoch_pretrain
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestone
        self.criterion_weight = criterion_weight
        self.scale_rec = 1.0
        self.scale_ad = 1.0
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.device = device
        self.print_batch_progress = print_batch_progress
        self.eps = 1e-6 # for numerical stability of loss function

        # Results
        self.pretrain_time = None
        self.pretrain_loss = None
        self.train_time = None
        self.train_loss = None

        self.valid_auc_rec = None
        self.valid_auc_ad = None
        self.valid_f1_rec = None
        self.valid_f1_ad = None
        self.valid_time = None
        self.valid_time = None
        self.valid_scores_rec = None
        self.valid_scores_ad = None

        self.test_auc_rec = None
        self.test_auc_ad = None
        self.test_f1_rec = None
        self.test_f1_ad = None
        self.test_time = None
        self.test_time = None
        self.test_scores_rec = None
        self.test_scores_ad = None

        # threhold to define if anomalous to maximize validation F1-score
        self.scores_threhold_rec = None
        self.scores_threhold_ad = None

    def pretrain(self, net, dataset):
        """
        Pretrain the AE for the joint DMSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is pretrained. It must return a tuple (image,
            |           label, mask, semi-supervized labels, idx).
            |---- net (nn.Module) The DMSAD to pretrain. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- net (nn.Module) The pretrained joint DMSAD.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net on device
        net = net.to(self.device)
        # set the network to provide only the reconstruction
        net.return_svdd_embed = False

        # define the reconstruvtion loss function
        loss_fn_rec = MaskedMSELoss()

        # define the optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Start training
        logger.info(' Start Pretraining the Autoencoder.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch = train_loader.__len__()

        for epoch in range(self.n_epoch_pretrain):
            net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get batch data
                input, _, mask, semi_label, _ = data
                input = input.to(self.device).float().requires_grad_(True)
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)

                # mask the input and keep only normal samples
                input = (input * mask)[semi_label != -1]
                mask = mask[semi_label != -1]

                # Update network parameters via backpropagation : forward + backward + optim
                optimizer.zero_grad()
                rec, _ = net(input)
                loss = loss_fn_rec(rec, input, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tBatch', Size=40, erase=True)

            # print epoch statstics
            logger.info(f'----| Epoch {epoch + 1:03}/{self.n_epoch_pretrain:03} '
                        f'| Pretrain Time {time.time() - epoch_start_time:.3f} [s] '
                        f'| Pretrain Loss {epoch_loss / n_batch:.6f} |')
            # store loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

        # End training
        self.pretrain_loss = epoch_loss_list
        self.pretrain_time = time.time() - start_time
        logger.info(f'---- Finished Pretraining the AutoEncoder in {self.pretrain_time:.3f} [s].')

        return net

    def train(self, net, dataset, valid_dataset=None):
        """
        Train the DMSAD on the provided dataset.
        ----------
        INPUT
            |---- net (nn.Module) The DMSAD to train. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervized labels.
            |---- valid_dataset (torch.utils.data.Dataset) the dataset on which
            |           to validate the model at each epoch. No validation is
            |           performed if not provided.
        OUTPUT
            |---- net (nn.Module) The pretrained joint DMSAD.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net on device
        net = net.to(self.device)
        # set the network to provide both the reconstruction and the embedding
        net.return_svdd_embed = True

        # Initialize the hyper-sphere centers by Kmeans
        if self.c is None:
            logger.info(' Initializing the hypersheres centers.')
            self.initialize_centers(train_loader, net)
            logger.info(f' {self.c.shape[0]} centers successfully initialized.')

        # define the reconstruvtion loss function
        loss_fn_rec = MaskedMSELoss()
        loss_fn_ad = DMSADLoss(self.eta, eps=self.eps)

        # Compute the scaling factors for the reconstruction and DMSAD losses
        logger.info(' Initializing the loss scale factors.')
        self.initialize_loss_scale_weight(train_loader, net, loss_fn_rec, loss_fn_ad)
        logger.info(f' reconstruction loss scale factor initialized to {self.scale_rec:.6f}')
        logger.info(f' MSAD embdeding loss scale factor initialized to {self.scale_ad:.6f}')

        # define the optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define the learning rate scheduler : 90% reduction at each steps
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info(' Start Training Jointly the DMSAD and the Autoencoder.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch = len(train_loader)

        for epoch in range(self.n_epoch):
            net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            n_k = torch.zeros(self.c.shape[0], device=self.device)

            for b, data in enumerate(train_loader):
                # get batch data
                input, _, mask, semi_label, _ = data
                input = input.to(self.device).float().requires_grad_(True)
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)

                # mask input
                input = input * mask

                # Update the network by backpropagation using the two losses.
                optimizer.zero_grad()
                rec, embed = net(input)
                # reconstruction loss only on normal sample (loss of zero for abnormal)
                rec = torch.where(semi_label.view(-1,1,1,1).expand(*input.shape) != 1, rec, input)
                loss_rec = self.scale_rec * self.criterion_weight[0] * loss_fn_rec(rec, input, mask)
                # DMSAD loss
                loss_ad = self.scale_ad * self.criterion_weight[1] * loss_fn_ad(embed, self.c, semi_label)
                # total loss
                loss = loss_rec + loss_ad
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # get the closest sphere and count the number of normal samples per sphere
                idx = torch.argmin(torch.norm(self.c.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=2), dim=1)
                for i in idx[semi_label != -1]:
                    n_k[i] += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=40, erase=True)

            # remove centers with less than alpha fraction of points <<<< TO CHECK or frac of max(n_k) ??
            self.c = self.c[n_k >= self.alpha * dataset.__len__()]

            # intermediate validation of the model if required
            valid_auc = ''
            if valid_dataset:
                auc_rec, auc_ad = self.evaluate(net, valid_dataset, mode='valid', final=False)
                valid_auc = f' Rec AUC {auc_rec:.3%} | MSAD AUC {auc_ad:.3%} |'

            # print epoch statstics
            logger.info(f'----| Epoch {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train Time {time.time() - epoch_start_time:.3f} [s] '
                        f'| Train Loss {epoch_loss / n_batch:.6f} '
                        f'| N sphere {self.c.shape[0]:03} |' + valid_auc)
            # store loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])

            # update learning rate if milestone is reached
            scheduler.step()
            if epoch + 1 in self.lr_milestone:
                logger.info(f'---- LR Scheduler : new learning rate {scheduler.get_lr()[0]:g}')

            # re-initialized loss scale factors after 3 epochs when the centers are more or less defined
            if epoch + 1 == 3:
                with torch.no_grad():
                    # Compute the scaling factors for the reconstruction and DMSAD losses
                    logger.info('---- Reinitializing the loss scale factors.')
                    self.initialize_loss_scale_weight(train_loader, net, loss_fn_rec, loss_fn_ad)
                    logger.info(f'---- reconstruction loss scale factor reinitialized to {self.scale_rec:.6f}')
                    logger.info(f'---- MSAD embdeding loss scale factor reinitialized to {self.scale_ad:.6f}')

        # End Training
        self.train_loss = epoch_loss_list
        self.train_time = time.time() - start_time
        logger.info(f'---- Finished jointly training the DMSAD and the Autoencoder in {self.train_time:.3f} [s].')

        return net

    def initialize_centers(self, loader, net, eps=0.1):
        """
        Initialize the multiple centers using the K-Means algorithm on the
        embedding of all the normal samples.
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
        # Get sample embedding
        repr = []
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data
                input, _, mask, semi_label, _ = data
                input = input.to(self.device).float()
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                # mask input and keep only normal samples
                input = (input * mask)[semi_label != -1]
                # get embdeding of batch
                _, embed = net(input)
                repr.append(embed)

                if self.print_batch_progress:
                    print_progessbar(b, loader.__len__(), Name='\t\tBatch', Size=40, erase=True)

            repr = torch.cat(repr, dim=0).cpu().numpy()

        # Apply Kmeans algorithm on embedding
        kmeans = KMeans(n_clusters=self.n_sphere_init).fit(repr)
        self.c = torch.tensor(kmeans.cluster_centers_).to(self.device)

        # check if c_i are epsilon too close to zero to avoid them to be trivialy matched to zero
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps

    def initialize_loss_scale_weight(self, loader, net, loss_fn_rec, loss_fn_ad):
        """
        Perform one forward pass to compute the reconstruction and embeding loss
        scaling factor so that the two loss have a magnitude of 1.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DMSAD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- loss_fn_rec (nn.Module) the reconstruction loss criterion.
            |---- loss_fn_ad (nn.Module) the MDAS embdeding loss criterion.
        OUTPUT
            |---- None
        """
        sumloss_rec, sumloss_ad = 0.0, 0.0
        n_batch = len(loader)

        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data
                input, _, mask, semi_label, _ = data
                input = input.to(self.device).float()
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                # mask input
                input = input * mask
                # Forward
                rec, embed = net(input)
                # Reconstruction loss only on normal sample (loss = 0 for abnormal)
                rec = torch.where(semi_label.view(-1,1,1,1).expand(*input.shape) != 1, rec, input)
                loss_rec = loss_fn_rec(rec, input, mask)
                sumloss_rec += loss_rec.item()
                # DMSAD loss
                loss_ad = loss_fn_ad(embed, self.c, semi_label)
                sumloss_ad += loss_ad.item()

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\tBatch', Size=40, erase=True)

            # get the scale factors
            self.scale_rec = 1 / (sumloss_rec / n_batch)
            self.scale_ad = 1 / (sumloss_ad / n_batch)

    def initialize_raidus(self, ):
        """
        compute radius as 1-gamma quatile of normal sample distance to center.

        for sample
            dist[sphere_idx] <- min(distances)

        r[sphere_idx] <- quantile(1-gamma, dist[sphere_idx])

        then score is ||net(x) - c_j||^2 - R_j^2 <--- negative if in, positive if out.
        """


    def evaluate(self, net, dataset, mode='test', final=False):
        """
        Evaluate the model with the given dataset.
        ----------
        INPUT
            |---- net (nn.Module) The DMSAD to validate. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image, a mask and
            |           semi-supervized labels.
            |---- mode (str) either 'valid' or 'test'. Define the evaluation mode.
            |           In 'valid' the evaluation can return the reconstruction
            |           and MSAD AUCs and compute the best threshold to maximize
            |           the F1-scores. In test mode the validation threshold is
            |           used to compute the F1-score.
            |---- final (bool) whether the call represents the final validation,
            |           in which case the validation results are saved. Only
            |           relevant if mode is 'valid'.
        OUTPUT
            |---- auc (tuple (reconstruction auc, ad auc)) the validation AUC for
            |           both scores are return only if final is False. Else None
            |           is return.
        """
        assert mode in ['valid','test'], f'Mode {mode} is not supported. Should be either "valid" or "test".'
        logger = logging.getLogger()

        # make the dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net on device
        net = net.to(self.device)
        # set the network to provide both the reconstruction and the embedding
        net.return_svdd_embed = True

        # define the two loss function
        loss_fn_rec = MaskedMSELoss(reduction='none') # no reduction to compute AD score for each sample
        loss_fn_ad = DMSADLoss(self.eta, self.eps)

        # Validate
        if final or mode == 'test':
            logger.info(f' Start Evaluating the jointly trained DMSAD and AutoEncoder in {mode} mode.')
        epoch_loss = 0.0
        n_batch = len(loader)
        start_time = time.time()
        idx_label_score_rec, idx_label_score_ad = [], [] # placeholder for scores

        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data on device
                input, label, mask, semi_label, idx = data
                input = input.to(self.device).float()
                label = label.to(self.device)
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                idx = idx.to(self.device)

                # mask input
                input = input * mask

                # compute the loss
                rec, embed = net(input)
                loss_rec = loss_fn_rec(rec, input, mask)
                loss_ad = loss_fn_ad(embed, self.c, semi_label)
                # get reconstruction anomaly scores : mean loss by sample
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim())))
                # get MSAD anomaly scores as the distance to closest center. c -> (N_sphere, Embed_dim) embed -> (Batch, Embed)
                ad_score, sphere_idx = torch.min(torch.sum((self.c.unsqueeze(0) - embed.unsqueeze(1))**2, dim=2), dim=1)

                # append scores to the placeholer lists
                idx_label_score_rec += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            rec_score.cpu().data.numpy().tolist()))
                idx_label_score_ad += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            ad_score.cpu().data.numpy().tolist(),
                                            sphere_idx.cpu().data.numpy().tolist()))

                # compute the overall loss
                loss = self.scale_rec * self.criterion_weight[0] * (torch.sum(loss_rec) / torch.sum(mask))
                loss += self.scale_ad * self.criterion_weight[1] * loss_ad
                epoch_loss += loss.item()

                if self.print_batch_progress:
                    print_progessbar(b, n_batch, Name='\t\t Evaluation Batch', Size=40, erase=True)

        # compute AUCs
        _, label, rec_score = zip(*idx_label_score_rec)
        label, rec_score = np.array(label), np.array(rec_score)
        auc_rec = roc_auc_score(label, rec_score)

        _, label, ad_score, _ = zip(*idx_label_score_ad)
        label, ad_score = np.array(label), np.array(ad_score)
        auc_ad = roc_auc_score(label, ad_score)

        if mode == 'valid':
            if final:
                # save results
                self.valid_time = time.time() - start_time
                self.valid_scores_rec = idx_label_score_rec
                self.valid_auc_rec = auc_rec
                self.scores_threhold_rec, self.valid_f1_rec = get_best_threshold(rec_score, label, metric=f1_score)
                self.valid_scores_ad = idx_label_score_ad
                self.valid_auc_ad = auc_ad
                self.scores_threhold_ad, self.valid_f1_ad = get_best_threshold(ad_score, label, metric=f1_score)
                # print infos
                logger.info(f'---- Validation Time: {self.valid_time:.3f} [s]')
                logger.info(f'---- Validation Loss: {epoch_loss / n_batch:.6f}')
                logger.info(f'---- Validation reconstruction AUC: {self.valid_auc_rec:.3%}')
                logger.info(f'---- Best Threshold for the reconstruction score maximizing F1-score: {self.scores_threhold_rec:.3f}')
                logger.info(f'---- Best F1-score on reconstruction score: {self.valid_f1_rec:.3%}')
                logger.info(f'---- Validation MSAD AUC: {self.valid_auc_ad:.3%}')
                logger.info(f'---- Best Threshold for the MSAD score maximizing F1-score: {self.scores_threhold_ad:.3f}')
                logger.info(f'---- Best F1-score on MSAD score: {self.valid_f1_ad:.3%}')
                logger.info('---- Finished validating the Joint DMSAD and AutoEncoder.\n')
            else:
                return auc_rec, auc_ad

        elif mode == 'test':
            # save results
            self.test_time = time.time() - start_time
            self.test_scores_rec = idx_label_score_rec
            self.test_auc_rec = auc_rec
            self.test_scores_ad = idx_label_score_ad
            self.test_auc_ad = auc_ad

            # print infos
            logger.info(f'---- Test Time: {self.test_time:.3f} [s]')
            logger.info(f'---- Test Loss: {epoch_loss / n_batch:.6f}')
            logger.info(f'---- Test reconstruction AUC: {self.test_auc_rec:.3%}')
            if self.scores_threhold_rec is not None:
                self.test_f1_rec = f1_score(label, np.where(rec_score > self.scores_threhold_rec, 1, 0))
                logger.info(f'---- Best F1-score on reconstruction score: {self.test_f1_rec:.3%}')
            logger.info(f'---- Test MSAD AUC: {self.test_auc_ad:.3%}')
            if self.scores_threhold_ad is not None:
                self.test_f1_ad = f1_score(label, np.where(ad_score > self.scores_threhold_ad, 1, 0))
                logger.info(f'---- Best F1-score on MSAD score: {self.test_f1_ad:.3%}')
            logger.info('---- Finished testing the Joint DMSAD and AutoEncoder.\n')
