import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.utils import print_progessbar, get_best_threshold

class DROCC_trainer:
    """
    Trainer for the DROCC following the paper of Goyal et al. (2020).
    """
    def __init__(self, r, gamma=0.5, mu=0.5, lr=1e-4, lr_adv=1e-2, lr_milestone=(),
                 weight_decay=1e-6, n_epoch=100, n_epoch_init=15, n_epoch_adv=15,
                 batch_size=16, device='cuda', n_jobs_dataloader=0, LFOC=False,
                 print_batch_progress=False):
        """
        Build a DROCC trainer.
        ----------
        INPUT
            |---- r (float) the radius to use.
            |---- gamma (float) the fraction of the radius defining the lower
            |           bound of the close adversarial samples layer.
            |---- mu (float) the adversarial loss weight in the total loss.
            |---- lr (float) the learning rate.
            |---- lr_adv (float) the learning rate for the adversarial search.
            |---- n_epoch (int) the number of epoch.
            |---- n_epoch_init (int) the number of epoch without adversarial search.
            |---- n_epoch_adv (int) the number of epoch for the gradient ascent.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to dispay the batch
            |           progress bar.
            |---- LFOC (bool) whether to use the LFOC implementation.
        OUTPUT
            |---- None
        """
        self.LFOC = LFOC # whether to use the DROCC-LF implementation

        self.r = r
        self.gamma = gamma
        self.mu = mu
        self.lr = lr
        self.lr_adv = lr_adv
        self.lr_milestone = lr_milestone
        self.weight_decay = weight_decay
        self.n_epoch = n_epoch
        self.n_epoch_init = n_epoch_init
        self.n_epoch_adv = n_epoch_adv
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.print_batch_progress = print_batch_progress

        # Results
        self.train_time = None
        self.train_loss = None

        self.valid_auc = None
        self.valid_f1 = None
        self.valid_time = None
        self.valid_scores = None

        self.test_auc = None
        self.test_f1 = None
        self.test_time = None
        self.test_scores = None

        # threhold to define if anomalous
        self.scores_threshold = None

    def train(self, dataset, net):
        """
        Train the DROCC network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is trained. It must return an image, a mask and
            |           semi-supervised labels.
            |---- net (nn.Module) The DROCC to train. The network should return
            |           the logit of the passed sample.
        OUTPUT
            |---- net (nn.Module) The trained DROCC.
        """
        logger = logging.getLogger()

        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, \
                                                   shuffle=True, num_workers=self.n_jobs_dataloader)
        # put net to device
        net = net.to(self.device)

        # loss function
        criterion = nn.BCEWithLogitsLoss()

        # define optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, gamma=0.1)

        # Start training
        logger.info('>>> Start Training the DROCC.')
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
                input = input.to(self.device).float()
                mask = mask.to(self.device)
                semi_label = semi_label.to(self.device)
                # get 'label' 0 = normal, 1 = abnormal
                semi_label = torch.where(semi_label != -1, torch.Tensor([0]).to(self.device), torch.Tensor([1]).to(self.device))
                # mask the input
                input = input * mask

                if epoch < self.n_epoch_init:
                    # initial steps without adversarial samples
                    input.requires_grad_(True)
                    logit = net(input).squeeze(dim=1)
                    loss = criterion(logit, semi_label)
                else:
                    # get adversarial samples
                    normal_input = input[semi_label == 0] # get normal input only for the adversarial search
                    adv_input = self.adversarial_search(normal_input, net)

                    # forward on both normal and adversarial samples
                    input.requires_grad_(True)
                    logit = net(input).squeeze(dim=1)
                    logit_adv = net(adv_input).squeeze(dim=1)
                    # loss of samples
                    loss_sample = criterion(logit, semi_label)
                    # loss of adversarial samples
                    loss_adv = criterion(logit_adv, torch.ones(adv_input.shape[0], device=self.device))
                    # weighted sum of normal and aversarial loss
                    loss = loss_sample + self.mu * loss_adv

                # Gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

                if self.print_batch_progress:
                    print_progessbar(b, n_batch_tot, Name='\t\tBatch', Size=20)

            # print epoch statistic
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epoch:03} | '
                        f'Train Time: {epoch_train_time:.3f} [s] '
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
        logger.info(f'>>> Training Time of DROCC: {self.train_time:.3f} [s]')
        logger.info('>>> Finished DROCC Training.\n')

        return net

    def adversarial_search2(self, x, net):
        """
        Perform an adversarial sample search by gradient ascent on the batch input.
        ----------
        INPUT
            |---- x (torch.Tensor) a batch of normal samples (B, (C,) H, W).
            |---- net (nn.Module) the network to fool.
        OUTPUT
            |---- x + h (torch.Tensor) the batch of adversarial samples.
        """
        # take the adversarial input around normal samples
        x_adv = x + torch.normal(0, 1, x.shape, device=self.device)
        x_adv = x_adv.detach().requires_grad_(True)
        # set optimizer for the the input h
        optimizer_adv = optim.SGD([x_adv], lr=self.lr_adv) # or SDG ???
        criterion = nn.BCEWithLogitsLoss()

        # get the sigmas for the LFOC manifold projection
        # target = 0 (normal samples)
        sigma = self.get_sigma(x, torch.zeros(x.shape[0], device=self.device), net) if self.LFOC else None

        # gradient ascent
        # net.eval() # the network parameters are not updated here
        for i in range(self.n_epoch_adv):
            # Update h to increase loss
            optimizer_adv.zero_grad()
            logit = net(x_adv).squeeze(dim=1)
            loss_h = criterion(logit, torch.ones(x_adv.shape[0], device=self.device)) # all adversarial samples are abnormal but the network should be fooled
            (-loss_h).backward()
            optimizer_adv.step()
            # Project h onto the the Ni(r)
            with torch.no_grad():
                h = self.project_on_manifold(x_adv - x, sigma)
                x_adv = x + h

        return x_adv.detach()

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
        # take the adversarial input around normal samples
        x = x.detach()
        h = torch.normal(0, 1, x.shape, device=self.device).requires_grad_(True)

        # set optimizer for the the input h
        optimizer_adv = optim.Adam([h], lr=self.lr_adv) # or SDG ???
        criterion = nn.BCEWithLogitsLoss()

        # get the sigmas for the LFOC manifold projection
        # target = 0 (normal samples)
        sigma = self.get_sigma(x, torch.zeros(x.shape[0], device=self.device), net) if self.LFOC else None

        # gradient ascent
        for i in range(self.n_epoch_adv):
            # Update h to increase loss
            optimizer_adv.zero_grad()
            logit = net(x + h).squeeze(dim=1)
            loss_h = criterion(logit, torch.ones(h.shape[0], device=self.device)) # all adversarial samples are abnormal but the network should be fooled
            (-loss_h).backward()
            optimizer_adv.step()
            # Project h onto the the Ni(r)
            with torch.no_grad():
                h = self.project_on_manifold(h, sigma)

        return (x + h).detach()

    def project_on_manifold(self, h, sigma):
        """
        Project the adversarial samples on the manifold.
        ----------
        INPUT
            |---- h (torch.Tensor) the difference between the normal and the
            |           adversarially generated samples (B, (C), H, W).
            |---- sigma (torch.Tensor) the gradient of the loss with respect to
            |           the input for LFOC. It has shape (B, (C), H, W).
        OUTPUT
            |---- h (torch.Tensor) the projected h.
        """
        if self.LFOC:
            # solve the 1D optimization problem described in Goyal et al. (2020)
            solver = ProjectionSolver(h, sigma, self.r, self.device)
            alpha = solver.solve()
            h = alpha * h
        else:
            # get the norm of h by batch
            norm_h = torch.sum(h**2, dim=tuple(range(1, h.dim())))
            # compute alpha in function of the value of the norm of h (by batch)
            alpha = torch.clamp(norm_h, self.gamma * self.r, self.r).to(self.device)
            # make use of broadcast to project h
            proj = (alpha / norm_h).view(-1, *[1]*(h.dim()-1))
            h = proj * h

        return h

    def get_sigma(self, data, target, net):
        """
        Compute the the gradient of the loss with respect to the input.
        ----------
        INPUT
            |---- data (torch.Tensor) a batch of data input.
            |---- target (torch.Tensor) the target labels associated with the inputs.
            |---- net (nn.Modules) the network to use.
        OUTPUT
            |---- sigma (torch.Tensor) the gradient of the network with respect
            |           to the input in absolute value divided by the norm.
        """
        criterion = nn.BCEWithLogitsLoss()
        grad_data, target = data.float().detach().requires_grad_(), target.float()
        # evaluate the logit with the data and compute the loss
        logit = net(grad_data).squeeze(dim=1)
        loss = criterion(logit, target)
        # get the derivative of loss compared to input data
        grad = torch.autograd.grad(loss, grad_data)[0]
        # normalize absolute value gradient by batch
        grad_norm = torch.sum(torch.abs(grad), dim=tuple(range(1, grad.dim())))
        sigma = torch.abs(grad) / grad_norm.view(-1, *[1]*(grad.dim()-1))

        return sigma

    def validate(self, dataset, net):
        """
        Validate the DROCC network on the provided dataset and find the best
        threshold on the score to maximize the f1-score.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is validated. It must return an image and
            |           semi-supervized labels.
            |---- net (nn.Module) The DROCC to validate. The network should return
            |           the logit of the passed sample.
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
        criterion = nn.BCEWithLogitsLoss()

        # Testing
        logger.info('>>> Start Validating of the DROCC.')
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
                input, label = input.to(self.device).float(), label.to(self.device).float()
                idx, mask = idx.to(self.device), mask.to(self.device)
                # mask input
                input = input * mask

                logit = net(input).squeeze(dim=1)
                loss = criterion(logit, label)
                # get the anomaly scores
                ad_score = torch.sigmoid(logit) # sigmoid of logit : should be high for abnormal (target = 1) and low for normal (target = 0)

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            ad_score.cpu().data.numpy().tolist()))

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
        logger.info('>>> Finished validating the DROCC.\n')

    def test(self, dataset, net):
        """
        Test the DROCC network on the provided dataset.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset on which the
            |           network is tested. It must return an image and
            |           semi-supervised labels.
            |---- net (nn.Module) The DROCC to test. The network should return
            |           the logit of the passed sample.
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
        criterion = nn.BCEWithLogitsLoss()

        # Testing
        logger.info('>>> Start Testing of the DROCC.')
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
                input, label = input.to(self.device).float(), label.to(self.device).float()
                idx, mask = idx.to(self.device), mask.to(self.device)
                # mask input
                input = input * mask

                logit = net(input).squeeze(dim=1)
                loss = criterion(logit, label)
                # get anomaly scores
                ad_score = torch.sigmoid(logit) # sigmoid of logit : should be high for abnormal (target = 1) and low for normal (target = 0)

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            label.cpu().data.numpy().tolist(),
                                            ad_score.cpu().data.numpy().tolist()))

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
        logger.info('>>> Finished testing the DROCC.\n')

class ProjectionSolver:
    """
    Numerical solver for the 1D non-convex optimization of DROCC-LF.
    """
    def __init__(self, sigma, h, radius, device='cuda', n_search=10):
        """
        Build a solver for the projection of DROCC-LF.
        ----------
        INPUT
            |---- sigma (torch.Tensor) the gradient of the loss with respect to
            |           the input. It has shape (B, (C), H, W).
            |---- h (torch.Tensor) the difference between the normal sample and
            |           the adversarialy generated one (B, (C), H, W)
            |---- radius (float) the radius around normal points.
            |---- device (str) the device to use ('cpu' or 'cuda').
            |---- n_search (int) the number of optimization iteration.
        OUTPUT
            |---- None
        """
        self.sigma = sigma
        self.h = h
        self.radius = radius
        self.device = device
        self.n_search = n_search
        # check where grid search will be done
        self.cond_init = self.check_cond_init()
        # get the minimal value of tau usable in grid search
        self.lower_tau = self.get_lower_tau()

    def check_cond_init(self):
        """
        Check the initial condition that the Mahalanobis distance of the
        difference is grater than r^2.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- condition (torch.Tensor) boolean tensor for each input of the batch.
        """
        # compute the Mahalanobis distance
        m_dist = torch.sum(self.sigma * self.h ** 2, dim=tuple(range(1, self.h.dim())))
        # return True where the Mahalanobis distance of h is greated than r^2
        return m_dist >= self.radius ** 2

    def get_lower_tau(self):
        """
        Get the lower bound for the tau parameters. It's given by -1/max(sigma).
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- low_tau (torch.Tensor) lower bound for each input of the batch.
        """
        # max of sigma by batch
        max_sigma, _ = torch.max(torch.flatten(self.sigma, start_dim=1), dim=1)
        #eps, _ = torch.min(torch.flatten(self.sigma, start_dim=1), dim=1) # ???
        low_tau = -1 / (max_sigma + 1e-10) #+ eps * 1e-4
        return low_tau#.detach().cpu().numpy()

    def eval_search_condition(self, tau, sigma_i, h_i):
        """
        Compute the condition upon which the metric to minimize is valid for a
        single sample.
        ----------
        INPUT
            |---- tau (float) the vlue of tau to use.
            |---- sigma_i (torch.Tensor) the sigma for the given sample (H, W, (C)).
            |---- h_i (torch.Tensor) the difference for the given sample (H, W, (C)).
        OUTPUT
            |---- condition (bool) whether the condition is fulfilled.
        """
        # check condition by sample and not by batch
        num = tau**2 * h_i**2 * sigma_i**3
        denom = (1 + tau * sigma_i)**2 + 1e-10
        val = torch.sum(num / denom)
        return val >= self.radius**2

    def eval_minimum(self, tau, sigma_i, h_i):
        """
        Compute the score to minimize for a single sample.
        ----------
        INPUT
            |---- tau (float) the vlue of tau to use.
            |---- sigma_i (torch.Tensor) the sigma for the given sample (H, W, (C)).
            |---- h_i (torch.Tensor) the difference for the given sample (H, W, (C)).
        OUTPUT
            |---- score (torch.Tensore) the scores to minimize for the sample.
        """
        num = tau**2 * h_i**2 * sigma_i**2
        denom = (1 + tau * sigma_i)**2 + 1e-10
        return torch.sum(num / denom)

    def solve(self):
        """
        Solve the 1D minimization problem by grid search for the samples in the batch.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- alpha (torch.Tensor) the coefficient to use to projects the
            |           samples onto the manifold : x_adv = x + alpha * h.
        """
        # placeholder for best taus
        best_tau = torch.zeros(self.h.shape[0], device=self.device)

        # for each sample individually
        for idx, cond_init in enumerate(self.cond_init):
            # if the grid search has to be done
            if cond_init:
                best_tau[idx] = 0
            else:
                min_val = np.inf
                for _ in range(self.n_search):
                    # pick a random value of tau
                    tau = torch.FloatTensor(1).uniform_(self.lower_tau[idx], 0)#np.random.uniform(low=self.lower_tau, high=0)
                    # check the score to minimize
                    if self.eval_search_condition(tau, self.sigma[idx], self.h[idx]):
                        val = self.eval_minimum(tau, self.sigma[idx], self.h[idx])
                    else:
                        val = torch.tensor(float('inf'))

                    if val < min_val:
                        min_val = val
                        best_tau[idx] = tau

        return 1 / (1 + best_tau.view(-1, *[1]*(self.sigma.dim()-1)) * self.sigma)
