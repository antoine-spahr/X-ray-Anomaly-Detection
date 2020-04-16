import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.utils import print_progessbar, get_best_threshold
from src.models.optim.CustomLosses import MaskedMSELoss

class ARAE_trainer:
    """
    Trainer for the ARAE following the paper of Salehi et al. (2020).
    """
    def __init__(self, gamma, epsilon, lr=1e-4, lr_adv=1e-2, lr_milestone=(),
                 weight_decay=1e-6, n_epoch=100, n_epoch_adv=15, use_PGD=True, batch_size=16,
                 device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Build a ARAE trainer.
        ----------
        INPUT
            |---- r (float) the radius to use.
            |---- gamma (float) the adversarial loss weight in the total loss.
            |---- epsilon (float) define l-inf bounds of the allowed adversarial
            |           perturbation of the normal inputs..
            |---- lr (float) the learning rate.
            |---- lr_adv (float) the learning rate for the adversarial search.
            |---- n_epoch (int) the number of epoch.
            |---- n_epoch_adv (int) the number of epoch for the gradient ascent.
            |---- use_PDG (bool) whether to use PGD or FGSM.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
        OUTPUT
            |---- None
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.lr_adv = lr_adv
        self.lr_milestone = lr_milestone
        self.weight_decay = weight_decay
        self.n_epoch = n_epoch
        self.n_epoch_adv = n_epoch_adv
        self.use_PGD = use_PGD
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.print_batch_progress = print_batch_progress

    def train(self, dataset, net):
        """
        Train the ARAE network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervised labels.
            |---- net (nn.Module) The ARAE to train. The network should be an
            |           autoencoder for which the forward pass returns both the
            |           reconstruction and the embedding of the input.
        OUTPUT
            |---- net (nn.Module) The trained ARAE.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net to device
        net = net.to(self.device)

        # define the criterions
        criterion_rec = MaskedMSELoss()
        criterion_lat = nn.MSELoss()

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('>>> Start Training the ARAE.')
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
                input =  input.to(self.device).float().requires_grad_(True)
                semi_label = semi_label.to(self.device)
                mask = mask.to(self.device)

                # mask input
                input = input * mask

                if self.use_PGD:
                    adv_input = self.adversarial_search(input, net)
                else:
                    adv_input = self.FGSM(input, net)
                    
                # pass the adversarial and normal samples through the network
                net.encoding_only = True
                _, lat = net(input)
                net.encoding_only = False
                rec_adv, lat_adv = net(adv_input)
                # compute the loss
                loss_rec = criterion_rec(adv_input, rec_adv, mask)
                loss_lat = criterion_lat(lat, lat_adv)
                loss = loss_rec + self.gamma * loss_lat

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

            # epoch statistic
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train Time: {epoch_train_time:.3f} [s] '
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
        logger.info(f'>>> Training Time of ARAE: {self.train_time:.3f} [s]')
        logger.info('>>> Finished ARAE Training.\n')

        return net

    def FGSM(self, x, net):
        """

        """
        # detach input
        x = x.detach()
        # set the network to give only the encoding (to speed up computation)
        net.encoding_only = True
        # define the loss fucntion
        loss_fn = nn.MSELoss()
        # initialize the perturbation in the range of [-epsilon, epsilon]
        delta = torch.zeros_like(x, device=self.device)
        delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)
        # forward/backward of normal and adversarial samples
        _, lat = net(x)
        _, lat_adv = net(x + delta)
        loss = loss_fn(lat, lat_adv)
        loss.backward()
        # steepest l-inf ascent and l-inf projection
        grad = delta.grad.detach()
        delta.data = delta + self.lr_adv * grad.sign()
        delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

        return (x + delta).detach()

    def adversarial_search(self, x, net):
        """
        Perform an adversarial sample search by gradient ascent on the batch input.
        ----------
        INPUT
            |---- x (torch.Tensor) a batch of normal samples (B, (C,) H, W).
            |---- net (nn.Module) the network to fool.
        OUTPUT
            |---- x + h (torch.Tensor) the batch of adversarial samples.
        """
        # detach input
        x = x.detach()
        # set the network to give only the encoding (to speed up computation)
        net.encoding_only = True
        # define the loss fucntion
        loss_fn = nn.MSELoss()
        # initialize the perturbation in the range of [-epsilon, epsilon]
        delta = torch.zeros_like(x, device=self.device)
        delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)

        for _ in range(self.n_epoch_adv):
            # forward of normal and adversarial samples
            _, lat = net(x)
            _, lat_adv = net(x + delta)
            loss = loss_fn(lat, lat_adv)
            loss.backward()
            # steepest l-inf ascent and l-inf projection
            grad = delta.grad.detach()
            delta.data = delta + self.lr_adv * grad.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            # reset the gradient
            delta.grad.zero_()

        # reset the AE net to return both latent and reconstruction
        net.encoding_only = False
        return (x + delta).detach()

    def validate(self, dataset, net):
        """
        Validate the ARAE network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image, a mask and
            |           semi-supervised labels.
            |---- net (nn.Module) The ARAE to validate. The network should be an
            |           autoencoder for which the forward pass returns both the
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

        # loss function
        criterion = MaskedMSELoss(reduction='none')

        # Testing
        logger.info('>>> Start Validating of the ARAE.')
        epoch_loss = 0.0
        n_batch = 0
        n_batch_tot = valid_loader.__len__()
        start_time = time.time()
        idx_label_score = []

        net.eval()
        with torch.no_grad():
            for b, data in enumerate(valid_loader):
                input, label, mask, _, idx = data
                # put data to device
                input = input.to(self.device).float()
                label = label.to(self.device).float()
                mask =  mask.to(self.device)
                idx = idx.to(self.device)
                # mask input
                input = input * mask

                rec, _ = net(input)
                loss = criterion(rec, input, mask)

                ad_score = torch.mean(loss, dim=tuple(range(1, rec.dim()))) # mean loss over batch

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            ad_score.cpu().data.numpy().tolist()))
                # compute the mean reconstruction loss
                loss = torch.sum(loss) / torch.sum(mask)
                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

        self.valid_time = time.time() - start_time
        self.valid_scores = idx_label_score
        _, label, ad_score = zip(*idx_label_score)
        label, ad_score = np.array(label), np.array(ad_score)
        self.valid_auc = roc_auc_score(label, ad_score)
        self.scores_threshold, self.valid_f1 = get_best_threshold(ad_score, label, metric=f1_score)

        # add info to logger
        logger.info(f'>>> Validation Time: {self.valid_time:.3f} [s]')
        logger.info(f'>>> Validation Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Validation AUC: {self.valid_auc:.3%}')
        logger.info(f'>>> Best Threshold for the score maximizing F1-score: {self.scores_threshold:.3f}')
        logger.info(f'>>> Best F1-score: {self.valid_f1:.3%}')
        logger.info('>>> Finished validating the ARAE.\n')

    def test(self, dataset, net):
        """
        Test the ARAE network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image, a mask and
            |           semi-supervised labels.
            |---- net (nn.Module) The ARAE to test. The network should be an
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

        # loss function
        criterion = MaskedMSELoss(reduction='none')

        # Testing
        logger.info('>>> Start Testing of the ARAE.')
        epoch_loss = 0.0
        n_batch = 0
        n_batch_tot = test_loader.__len__()
        start_time = time.time()
        idx_label_score = []

        net.eval()
        with torch.no_grad():
            for b, data in enumerate(test_loader):
                input, label, mask, _, idx = data
                # put data to device
                input = input.to(self.device).float()
                label = label.to(self.device).float()
                mask =  mask.to(self.device)
                idx = idx.to(self.device)
                # mask input
                input = input * mask

                rec, _ = net(input)
                loss = criterion(rec, input, mask)

                ad_score = torch.mean(loss, dim=tuple(range(1, rec.dim()))) # mean loss over batch

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            ad_score.cpu().data.numpy().tolist()))
                # compute the mean reconstruction loss
                loss = torch.sum(loss) / torch.sum(mask)
                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score
        _, label, ad_score = zip(*idx_label_score)
        label, ad_score = np.array(label), np.array(ad_score)
        self.test_auc = roc_auc_score(label, ad_score)
        self.test_f1 = f1_score(label, np.where(ad_score > self.scores_threshold, 1, 0))

        # add info to logger
        logger.info(f'>>> Testing Time: {self.test_time:.3f} [s]')
        logger.info(f'>>> Test Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Test AUC: {self.test_auc:.3%}')
        logger.info(f'>>> Test F1-score: {self.test_f1:.3%}')
        logger.info('>>> Finished testing the ARAE.\n')
