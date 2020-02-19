import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score

from src.models.optim.CustomLosses import DeepSADLoss
from src.utils.utils import print_progessbar

class DeepSADTrainer:
    """
    Trainer for the DeepSAD.
    """
    def __init__(self, c, eta, lr=0.0001, n_epoch=150, lr_milestone=(), batch_size=64,
                 weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Constructor of the DeepSAD trainer.
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

        # DeepSAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, net):
        """
        Train the DeepSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSAD to train.
        OUTPUT
            |---- net (nn.Module) The trained DeepSAD.
        """
        logger = logging.getLogger()

        # make the train dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)

        # put net to device
        net = net.to(self.device)

        # initialize hypersphere center
        if self.c is None:
            logger.info('>>> Initializing the hypersphere center.')
            self.c = self.initialize_hypersphere_center(train_loader, net)
            logger.info('>>> Center succesfully initialized.')

        # define loss criterion
        criterion = DeepSADLoss(self.c, self.eta, eps=self.eps)

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('>>> Start Training of the DeepSAD.')
        start_time = time.time()
        # set network in train mode
        net.train()

        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            n_batch = 0
            epoch_start_time = time.time()
            for b, data in enumerate(train_loader):
                # get input and semi-supervized labels
                input, _, _, semi_label, _ = data
                input.requires_grad = True
                # put them to device
                input, semi_label = input.to(self.device).float(), semi_label.to(self.device)
                # zero the network's gradients
                optimizer.zero_grad()

                # optimize by backpropagation
                output = net(input)
                #loss = criterion(output, semi_label)
                dist = torch.sum((output - self.c) ** 2, dim=1)
                losses = torch.where(semi_label == 0, dist, self.eta * ((dist + self.eps) ** semi_label.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, train_loader.__len__(), Name='Batch', Size=20)

            # log the epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch:03} | Train Time: {epoch_train_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} |')

            # update scheduler
            scheduler.step()
            if epoch in self.lr_milestone:
                logger.info('>>> LR Scheduler : new learning rate %g' % float(scheduler.get_lr()[0]))

        # End training
        self.train_time = time.time() - start_time
        logger.info(f'>>> Training of DeepSAD Time: {self.train_time:.3f} [s]')
        logger.info('>>> Finished DeepSAD Training.\n')

        return net

    def test(self, dataset, net):
        """
        Test the DeepSAD network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DeepSAD network to test.
        OUTPUT
            |---- None
        """
        logger = logging.getLogger()

        # make the train dataloader
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                  shuffle=True, num_workers=self.n_jobs_dataloader)

        # put net to device
        net = net.to(self.device)

        criterion = DeepSADLoss(self.c, self.eta, eps=self.eps)

        # Testing
        logger.info('>>> Start Testing of the DeepSAD')
        epoch_loss = 0.0
        n_batch = 0
        start_time = time.time()
        idx_label_score = []
        # set the network in evaluation mode
        net.eval()

        with torch.no_grad():
            for b, data in enumerate(test_loader):
                input, label, _, semi_label, idx = data
                # put inputs to device
                input, label = input.to(self.device).float(), label.to(self.device)
                semi_label, idx = semi_label.to(self.device), idx.to(self.device)

                # compute output loss
                output = net(input)
                loss = criterion(output, semi_label)
                score = torch.sum((output - self.c) ** 2, dim=1) # score is the distance (large distances highlight anomalies)

                # append scores and label
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, test_loader.__len__(), Name='Batch', Size=20)

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC : if AE is good a high reconstruction loss highlights the presence of an anomaly on the image
        _, label, score = zip(*idx_label_score)
        label, score = np.array(label), np.array(score)
        self.test_auc = roc_auc_score(label, score)

        # add info to logger
        logger.info(f'>>> Test Time: {self.test_time:.3f} [s]')
        logger.info(f'>>> Test Loss: {epoch_loss / n_batch:.6f}')
        logger.info(f'>>> Test AUC: {self.test_auc:.3%}')
        logger.info('>>> Finished Testing the DeepSAD.\n')

    def initialize_hypersphere_center(self, loader, net, eps=0.1):
        """
        Initialize the hypersphere center as the mean output of the network over
        one forward pass.
        ----------
        INPUT
            |---- loader (torch.utils.data.DataLoader) the loader of the data.
            |---- net (nn.Module) the DeepSAD network. The output must be a vector
            |           embedding of the input.
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
                output = net(input)
                n_sample += output.shape[0]
                c += torch.sum(output, dim=0)

                if self.print_batch_progress:
                    print_progessbar(b, loader.__len__(), Name='Batch', Size=20)

        # take the mean of accumulated c
        c /= n_sample
        # check if c_i are epsilon too close to zero to avoid them to be rivialy matched to zero
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c > 0)] = eps

        return c
