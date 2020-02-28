import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score

from src.models.optim.CustomLosses import DeepSADLoss, MaskedMSELoss
from src.utils.utils import print_progessbar, get_best_threshold

class DeepSAD_Joint_trainer:
    """
    Trainer for the DeepSAD together with the autoencoder (joint training).
    """
    def __init__(self, c, eta, lr=0.0001, n_epoch=150, lr_milestone=(), batch_size=64,
                 weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False,
                 criterion_weight=(0.5,0.5)):
        """
        Build a Joint DeepSAD and AutoEncoder trainer.
        ----------
        INPUT
            |---- c (torch.Tensor) the hypersphere center.
            |---- eta (float) the deep SAD parameter weighting the importance of
            |           unkonwn/known sample in learning.
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
            |---- criterion_weight (tuple (weight reconstruction, weight SAD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           Deep SAD).
        OUTPUT
            |---- None
        """
        # learning parameters
        self.lr = lr
        self.n_epoch = n_epoch
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.print_batch_progress = print_batch_progress
        self.criterion_weight = criterion_weight

        # DeepSAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.train_loss = None

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

    def train(self, dataset, net):
        """
        Train the joint DeepSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSAD to train. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- net (nn.Module) The trained joint DeepSAD.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net to device
        net = net.to(self.device)

        # initialize hypersphere center
        if self.c is None:
            logger.info('>>> Initializing the hypersphere center.')
            self.c = self.initialize_hypersphere_center(train_loader, net)
            logger.info('>>> Center succesfully initialized.')

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss()
        criterion_ad = DeepSADLoss(self.c, self.eta, eps=self.eps)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('>>> Start Training the Joint DeepSAD and Autoencoder.')
        start_time = time.time()
        epoch_loss_list = []
        n_batch_tot = train_loader.__len__()
        # set network in train mode
        net.train()
        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            n_batch = 0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                input, _, mask, semi_label, _ = data
                # put inputs to device
                input, mask, semi_label = input.to(self.device).float(), mask.to(self.device), semi_label.to(self.device)
                input.requires_grad = True

                # zeros the gradient
                optimizer.zero_grad()

                # Update network parameters by backpropagation on the two criterion
                rec, embed = net(input)
                loss_rec = criterion_rec(rec, input, mask)
                loss_ad = criterion_ad(embed, semi_label)
                loss = self.criterion_weight[0]*loss_rec + self.criterion_weight[1]*loss_ad
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

            # epoch statistic
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch:03} | Train Time: {epoch_train_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} |')

            epoch_loss_list += [[epoch+1, epoch_loss/n_batch]]

            # apply the scheduler step
            scheduler.step()
            if epoch in self.lr_milestone:
                logger.info('>>> LR Scheduler : new learning rate %g' % float(scheduler.get_lr()[0]))

        # End training
        self.train_loss = epoch_loss_list
        self.train_time = time.time() - start_time
        logger.info(f'>>> Training of Joint DeepSAD and AutoEncoder Time: {self.train_time:.3f} [s]')
        logger.info('>>> Finished of Joint DeepSAD and AutoEncoder Training.\n')

        return net

    def validate(self, dataset, net):
        """
        Validate the joint DeepSAD network on the provided dataset and find the
        best threshold on the score to maximize the f1-score.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSAD to validate. The network should be
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

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss(reduction='none')
        criterion_ad = DeepSADLoss(self.c, self.eta, eps=self.eps)

        # Testing
        logger.info('>>> Start Validating of the joint DeepSAD and AutoEncoder.')
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
                # compute loss
                rec, embed = net(input)
                loss_rec = criterion_rec(rec, input, mask)
                loss_ad = criterion_ad(embed, semi_label)
                # compute anomaly scores
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch
                ad_score = torch.sum((embed - self.c) ** 2, dim=1) # score is the distance (large distances highlight anomalies)
                # compute overall loss
                mean_loss_rec = torch.sum(loss_rec) / torch.sum(mask)
                loss = self.criterion_weight[0]*mean_loss_rec + self.criterion_weight[1]*loss_ad

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
        # Compute AUC : if AE is good a high reconstruction loss highlights the presence of an anomaly on the image
        # Compute AUC : if DeepSAD is good a distance highlights the presence of an anomaly on the image
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
        logger.info(f'>>> Validation DeepSAD AUC: {self.valid_auc_ad:.3%}')
        logger.info(f'>>> Best Threshold for the DeepSAD score maximizing F1-score: {self.scores_threhold_ad:.3f}')
        logger.info(f'>>> Best F1-score on DeepSAD score: {self.valid_f1_ad:.3%}')
        logger.info('>>> Finished validating the Joint DeepSAD and AutoEncoder.\n')

    def test(self, dataset, net):
        """
        Test the joint DeepSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSAD to test. The network should be an
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

        # define the two criterion for Anomaly detection and reconstruction
        criterion_rec = MaskedMSELoss(reduction='none')
        criterion_ad = DeepSADLoss(self.c, self.eta, eps=self.eps)

        # Testing
        logger.info('>>> Start Testing the joint DeepSAD and AutoEncoder.')
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
                # compute loss
                rec, embed = net(input)
                loss_rec = criterion_rec(rec, input, mask)
                loss_ad = criterion_ad(embed, semi_label)
                # compute anomaly scores
                rec_score = torch.mean(loss_rec, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch
                ad_score = torch.sum((embed - self.c) ** 2, dim=1) # score is the distance (large distances highlight anomalies)
                # get overall loss
                mean_loss_rec = torch.sum(loss_rec) / torch.sum(mask)
                loss = self.criterion_weight[0]*mean_loss_rec + self.criterion_weight[1]*loss_ad

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
        # Compute AUC : if AE is good a high reconstruction loss highlights the presence of an anomaly on the image
        # Compute AUC : if DeepSAD is good a distance highlights the presence of an anomaly on the image
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
        logger.info(f'>>> Test F1-score on DeepSAD score: {self.test_f1_ad:.3%}')
        logger.info('>>> Finished Testing the Joint DeepSAD and AutoEncoder.\n')

    def initialize_hypersphere_center(self, loader, net, eps=0.1):
        """
        Initialize the hypersphere center as the mean output of the network over
        one forward pass.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DeepSAD network. The output must be a vector
            |           embedding of the input. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
            |---- eps (float) the epsilon representing the minimum value of the
            |           component of the center.
        OUTPUT
            |---- c (torch.Tensor) the initialized center.
        """
        n_sample = 0
        c = torch.zeros(net.embed_dim, device=self.device)
        # get the output of all samples and accumulate them
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, _, _, _, _ = data
                input = input.to(self.device).float()
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
