import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score

from src.models.optim.CustomLosses import MaskedMSELoss
from src.utils.utils import print_progessbar, get_best_threshold

class AutoEncoderTrainer:
    """
    Trainer for the AutoEncoder.
    """
    def __init__(self, lr=0.0001, n_epoch=150, lr_milestone=(), batch_size=64,
                 weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Constructor of the AutoEncoder trainer.
        ----------
        INPUT
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
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
        # set up attributes
        self.lr = lr
        self.n_epoch = n_epoch
        self.lr_milestone = lr_milestone
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.print_batch_progress = print_batch_progress
        # output attributes
        self.train_time = None
        self.train_loss = None

        self.valid_time = None
        self.valid_auc = None
        self.valid_f1 = None
        self.valid_scores = None

        self.test_time = None
        self.test_auc = None
        self.test_f1 = None
        self.test_scores = None

        # threshold to define anomalous
        self.scores_threhold = None

    def train(self, dataset, ae_net):
        """
        Train the autoencoder network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image and a mask
            |           of where the loss is to be computed.
            |---- ae_net (nn.Module) The autoencoder to train.
        OUTPUT
            |---- ae_net (nn.Module) The trained autoencoder.
        """
        logger = logging.getLogger()

        # make train dataloader using image and mask
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.n_jobs_dataloader)

        # MSE loss without reduction --> MSE loss for each output pixels
        criterion = MaskedMSELoss()

        # set to device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # set optimizer
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # set the learning rate scheduler (multiple phase learning)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Training
        logger.info('>>> Start Training the AutoEncoder.')
        start_time = time.time()
        epoch_loss_list = []
        # set the network in train mode
        ae_net.train()

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            n_batch = 0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                input, _, mask, _, _ = data
                # put inputs to device
                input, mask = input.to(self.device).float(), mask.to(self.device)

                # zero the network gradients
                optimizer.zero_grad()

                # Update network paramters by backpropagation by considering only the loss on the mask
                rec = ae_net(input)
                loss = criterion(rec, input, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, train_loader.__len__(), Name='\t\tBatch', Size=20)

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
        logger.info(f'>>> Training of AutoEncoder Time: {self.train_time:.3f} [s]')
        logger.info('>>> Finished AutoEncoder Training.\n')

        return ae_net

    def validate(self, dataset, ae_net):
        """
        Validate the autoencoder network on the provided dataset and find the
        best threshold on the score to maximize the f1-score.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image and a mask
            |           of where the loss is to be computed.
            |---- ae_net (nn.Module) The autoencoder network to validate.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()

        # make test dataloader using image and mask
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.n_jobs_dataloader)

        # MSE loss without reduction --> MSE loss for each output pixels
        criterion = MaskedMSELoss(reduction='none')

        # set to device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('>>> Start Validating the AutoEncoder.')
        epoch_loss = 0.0
        n_batch = 0
        start_time = time.time()
        idx_label_score = []
        # put network in evaluation mode
        ae_net.eval()

        with torch.no_grad():
            for b, data in enumerate(valid_loader):
                input, label, mask, _, idx = data
                # put inputs to device
                input, label = input.to(self.device).float(), label.to(self.device)
                mask, idx = mask.to(self.device), idx.to(self.device)

                rec = ae_net(input)
                rec_loss = criterion(rec, input, mask)
                score = torch.mean(rec_loss, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch

                # append scores and label
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist()))
                # overall batch loss
                loss = torch.sum(rec_loss) / torch.sum(mask)
                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, valid_loader.__len__(), Name='\t\tBatch', Size=20)

        self.valid_time = time.time() - start_time
        self.valid_scores = idx_label_score

        # Compute AUC : if AE is good a high reconstruction loss highlights the presence of an anomaly on the image
        _, label, score = zip(*idx_label_score)
        label, score = np.array(label), np.array(score)
        self.valid_auc = roc_auc_score(label, score)
        self.scores_threhold, self.valid_f1 = get_best_threshold(score, label, metric=f1_score)

        # add info to logger
        logger.info(f'>>> Validation Time: {self.valid_time:.3f} [s]')
        logger.info(f'>>> Validation Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Validation AUC: {self.valid_auc:.3%}')
        logger.info(f'>>> Best Threshold maximizing the F1-score: {self.scores_threhold:.3f}')
        logger.info(f'>>> Best Validation F1-score: {self.valid_f1:.3%}')
        logger.info('>>> Finished Validating the AutoEncoder.\n')

    def test(self, dataset, ae_net):
        """
        Test the autoencoder network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image and a mask
            |           of where the loss is to be computed.
            |---- ae_net (nn.Module) The autoencoder network to test.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()

        # make test dataloader using image and mask
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.n_jobs_dataloader)

        # MSE loss without reduction --> MSE loss for each output pixels
        criterion = MaskedMSELoss(reduction='none')

        # set to device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('>>> Start Testing the AutoEncoder.')
        epoch_loss = 0.0
        n_batch = 0
        start_time = time.time()
        idx_label_score = []
        # put network in evaluation mode
        ae_net.eval()

        with torch.no_grad():
            for b, data in enumerate(test_loader):
                input, label, mask, _, idx = data
                # put inputs to device
                input, label = input.to(self.device).float(), label.to(self.device)
                mask, idx = mask.to(self.device), idx.to(self.device)

                rec = ae_net(input)
                rec_loss = criterion(rec, input, mask)
                score = torch.mean(rec_loss, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch

                # append scores and label
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist()))
                # overall batch loss
                loss = torch.sum(rec_loss) / torch.sum(mask)
                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, test_loader.__len__(), Name='\t\tBatch', Size=20)

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC : if AE is good a high reconstruction loss highlights the presence of an anomaly on the image
        _, label, score = zip(*idx_label_score)
        label, score = np.array(label), np.array(score)
        self.test_auc = roc_auc_score(label, score)
        self.test_f1 = f1_score(label, np.where(score > self.scores_threhold, 1, 0))

        # add info to logger
        logger.info(f'>>> Test Time: {self.test_time:.3f} [s]')
        logger.info(f'>>> Test Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Test AUC: {self.test_auc:.3%}')
        logger.info(f'>>> Test F1-score: {self.test_f1:.3%}')
        logger.info('>>> Finished Testing the AutoEncoder.\n')
