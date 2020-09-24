import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """
    Compute the MSE loss only on the masked region.
    """
    def __init__(self, reduction='mean'):
        """
        Loss Constructor.
        ----------
        INPUT
            |---- reduction (str) the reduction to use on the loss. ONLY 'mean' or 'none'.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input, target, mask):
        """
        Forard pass of the loss. The loss is computed only where the wask is non-null.
        ----------
        INPUT
            |---- input (torch.Tensor) the input tensor.
            |---- target (torch.Tensor) the target tensor.
            |---- mask (torch.Tensor) the binary mask defining where the loss is
            |           computed (loss computed where mask != 0).
        OUTPUT
            |---- loss (torch.Tensor) the masked MSE loss.
        """
        # compute loss where mask = 1
        loss = self.criterion(input * mask, target * mask)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / torch.sum(mask)
        return loss

class DeepSADLoss(nn.Module):
    """
    Implementation of the DeepSAD loss proposed by Lukas Ruff et al. (2019)
    """
    def __init__(self, c, eta, eps=1e-6):
        """
        Constructor of the DeepSAD loss.
        ----------
        INPUT
            |---- c (torch.Tensor) the center of the hypersphere as a multidimensional vector.
            |---- eta (float) control the importance given to known or unknonw
            |           samples. 1.0 gives equal weights, <1.0 gives more weight
            |           to the unknown samples, >1.0 gives more weight to the
            |           known samples.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.c = c
        self.eta = eta
        self.eps = eps

    def forward(self, input, semi_target):
        """
        Forward pass of the DeepSAD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere
            |           center. (must thus have the same dimension (B x c.dim)).
            |---- semi_target (torch.Tensor) the semi-supervized label (0 -> unknown ;
            |           1 -> known normal ; -1 -> knonw abnormal)
        OUTPUT
            |---- loss (torch.Tensor) the DeepSAD loss.
        """
        # distance between center c and the input
        #dist = torch.sum((self.c - input)**2, dim=1)
        dist = torch.norm(self.c - input, p=2, dim=1)
        # compute the loss depening on the semi-supervized label
        # keep distance if semi-label is 0 or 1 (normal sample or unknonw (assumed) normal)
        # inverse distance if semi-label = -1 (known abnormal)
        losses = torch.where(semi_target == 0, dist**2, self.eta * ((dist**2 + self.eps) ** semi_target.float()))
        loss = torch.mean(losses)
        return loss

class DeepSADLossSubspace(nn.Module):
    """
    Implementation of the DeepSAD loss proposed by Lukas Ruff et al. (2019) but
    with the distance of the point projected to the subspace of training samples
    rather than the hypersphere. It follows the mathematical derivation proposed
    by Arnout Devos et al. (2019).
    """
    def __init__(self, P, eta, eps=1e-6):
        """
        Constructor of the DeepSAD loss Subspace.
        ----------
        INPUT
            |---- P (torch.Tensor) The projection matrix to the subspace of normal
            |            sample. P is a MxM matrix where M is the embedding dimension.
            |---- eta (float) control the importance given to known or unknonw
            |           samples. 1.0 gives equal weights, <1.0 gives more weight
            |           to the unknown samples, >1.0 gives more weight to the
            |           known samples.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.P = P
        self.eta = eta
        self.eps = eps

    def forward(self, input, semi_target):
        """
        Forward pass of the DeepSAD loss Subspace.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to project onto the subspace.
            |           (must thus have the same dimension (B x embed.dim)).
            |---- semi_target (torch.Tensor) the semi-supervized label (0 -> unknown ;
            |           1 -> known normal ; -1 -> knonw abnormal)
        OUTPUT
            |---- loss (torch.Tensor) the DeepSAD loss Subspace.
        """
        # distance between center c and the input (tranpose to manage the batch dimension)
        dist = torch.sum((input - torch.matmul(self.P, input.transpose(0,1)).transpose(0,1))**2, dim=1)
        # compute the loss depening on the semi-supervized label
        # keep distance if semi-label is 0 or 1 (normal sample or unknonw (assumed) normal)
        # inverse distance if semi-label = -1 (known abnormal)
        losses = torch.where(semi_target == 0, dist, self.eta * ((dist + self.eps) ** semi_target.float()))
        loss = torch.mean(losses)
        return loss

class DeepSVDDLoss(nn.Module):
    """
    Implementation of the DeepSVDD loss proposed by Lukas Ruff et al. (2019)
    """
    def __init__(self, c, nu, eps=1e-6, soft_boundary=False):
        """
        Constructor of the DeepSVDD loss.
        ----------
        INPUT
            |---- c (torch.Tensor) the center of the hypersphere as a multidimensional vector.
            |---- nu (float) a priory fraction of outliers.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
            |---- soft_boundary (bool) whether to use a soft boundary
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.c = c
        self.nu = nu
        self.eps = eps
        self.soft_boundary = soft_boundary

    def forward(self, input, R):
        """
        Forward pass of the DSVDD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere
            |           center. (must thus have the same dimension (B x c.dim)).
            |---- R (flaot) radius for the soft boundary.
        OUTPUT
            |---- loss (torch.Tensor) the DeepSVDD loss.
        """
        # distance between center c and the input
        dist = torch.sum((self.c - input)**2, dim=1)
        # compute the loss
        if self.soft_boundary:
            scores = dist - R ** 2
            loss = R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        return loss

class DeepSVDDLossSubspace(nn.Module):
    """
    Implementation of the DeepSVDD loss proposed by Lukas Ruff et al. (2019) but
    with the distance of the point projected to the subspace of training samples
    rather than the hypersphere. It follows the mathematical derivation proposed
    by Arnout Devos et al. (2019).
    """
    def __init__(self, P, nu, eps=1e-6, soft_boundary=False):
        """
        Constructor of the DeepSVDD loss Subspace.
        ----------
        INPUT
            |---- P (torch.Tensor) The projection matrix to the subspace of normal
            |            sample. P is a MxM matrix where M is the embedding dimension.
            |---- nu (float) a priory fraction of outliers.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
            |---- soft_boundary (bool) whether to use a soft boundary.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.P = P
        self.nu = nu
        self.eps = eps

    def forward(self, input, R):
        """
        Forward pass of the DeepSVDD loss Subspace.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to project onto the subspace.
            |           (must thus have the same dimension (B x embed.dim)).
            |---- R (flaot) radius for the soft boundary.
        OUTPUT
            |---- loss (torch.Tensor) the DeepSVDD loss Subspace.
        """
        # distance between center c and the input (tranpose to manage the batch dimension)
        dist = torch.sum((input - torch.matmul(self.P, input.transpose(0,1)).transpose(0,1))**2, dim=1)
        # compute the loss
        if self.soft_boundary:
            scores = dist - R ** 2
            loss = R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        return loss

class DMSVDDLoss(nn.Module):
    """
    Implementation of the DMSVDD loss proposed by Ghafoori et al. (2020).
    """
    def __init__(self, nu, eps=1e-6, soft_boundary=False):
        """
        Constructor of the DMSVDD loss.
        ----------
        INPUT
            |---- nu (float) a priory fraction of outliers.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
            |---- soft_boundary (bool) whether to use a soft boundary
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.nu = nu
        self.eps = eps
        self.soft_boundary = soft_boundary

    def forward(self, input, c, R):
        """
        Forward pass of the DMSVDD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere.
            |           center. (must thus have the same dimension (B x c.dim)).
            |---- c (torch.Tensor) the centers of the hyperspheres as a multidimensional matrix (Centers x Embdeding).
            |---- R (float) the radius for the soft boundary (length = nbr of centers).
        OUTPUT
            |---- loss (torch.Tensor) the DMSVDD loss.
        """
        # distance between the input and the closest center
        dist, idx = torch.min(torch.sum((c.unsqueeze(0) - input.unsqueeze(1))**2, dim=2), dim=1) # dist and idx by batch

        # compute the loss
        if self.soft_boundary:
            #scores = dist - R**2
            scores = dist - torch.stack([R[i] ** 2 for i in idx], dim=0)
            #loss = 1/R.shape[0] * torch.sum(R ** 2) + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            loss = torch.mean(R ** 2) + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        return loss

class DMSADLoss(nn.Module):
    """
    Implementation of the DMSAD loss inspired by Ghafoori et al. (2020) and Ruff
    et al. (2020)
    """
    def __init__(self, eta, eps=1e-6):
        """
        Constructor of the DMSAD loss.
        ----------
        INPUT
            |---- eta (float) control the importance given to known or unknonw
            |           samples. 1.0 gives equal weights, <1.0 gives more weight
            |           to the unknown samples, >1.0 gives more weight to the
            |           known samples.
            |---- eps (float) epsilon to ensure numerical stability in the
            |           inverse distance.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.eta = eta
        self.eps = eps

    def forward(self, input, c, semi_target):
        """
        Forward pass of the DMSAD loss.
        ----------
        INPUT
            |---- input (torch.Tensor) the point to compare to the hypershere.
            |           center.
            |---- c (torch.Tensor) the centers of the hyperspheres as a multidimensional matrix (Centers x Embdeding).
            |---- semi_target (torch.Tensor) the semi-supervized label (0 -> unknown ;
            |           1 -> known normal ; -1 -> knonw abnormal)
        OUTPUT
            |---- loss (torch.Tensor) the DMSAD loss.
        """
        # distance between the input and the closest center
        dist, _ = torch.min(torch.norm(c.unsqueeze(0) - input.unsqueeze(1), p=2, dim=2), dim=1) # dist and idx by batch
        # compute the loss
        losses = torch.where(semi_target == 0, dist**2, self.eta * ((dist**2 + self.eps) ** semi_target.float()))
        # losses = torch.where(semi_target == 0, dist**2,
        #                                        self.eta * torch.where(semi_target == -1, -torch.log(1-torch.exp(-(dist**2 + self.eps))),
        #                                                               dist**2)) # -torch.log(1-torch.exp(-dist**2)) # 1/(dist**2 + self.eps)
        loss = torch.mean(losses)
        return loss

class NT_Xent_loss(nn.Module):
    """
    Normalized temperature scaled cross-entropy loss.
    (https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py)
    """
    def __init__(self, tau, batch_size, device='cuda'):
        """

        """
        nn.Module.__init__(self)
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.negative_mask = self.get_neg_mask(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def get_neg_mask(self, batch_size):
        """

        """
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """

        """
        # concat both represenation to get a (2*Batch x Embed)
        p = torch.cat((z_i, z_j), dim=0)
        # Compute the similarity matrix between elements --> (2Batch x 2Batch)
        sim = self.similarity_f(p.unsqueeze(1), p.unsqueeze(0)) / self.tau
        # Get positive pair similarity --> diag of upper right and lower left quarter of sim
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # get positive and negative vector for the two batches
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*self.batch_size, 1)
        negative_samples = sim[self.negative_mask].reshape(2*self.batch_size, -1)
        # generate lables (the first element represent the correct pair --> zero is correct label)
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # Compute the CE loss
        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.batch_size)
        return loss

class SupervisedContrastiveLoss(nn.Module):
    """
    Define the Supervised Contrastive Loss as a Pytorch Module.
    """
    def __init__(self, tau, batch_size, y_list='all', device='cuda'):
        """
        Initialize a Supervised Contrastive Loss Module.
        ----------
        INPUT
            |---- tau (float) the temperature hyperparameter.
            |---- y_list (list of int) the list of class to conisder for positive.
            |           Default is using all classes.
            |---- batch_size (int) the batch_size used.
            |---- device (str) the device to use.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.tau = tau
        self.y_list = y_list
        self.batch_size = batch_size
        self.device=device

    def forward(self, z_i, z_j, y):
        """
        Forward pass of the Supervised Contrastive Loss.
        ----------
        INPUT
            |---- z_i (torch.Tensor) the representation of the 1st batch of augmented
            |           input with dimension (Batch x Embed).
            |---- z_j (torch.Tensor) the representation of the 2st batch of augmented
            |           input with dimension (Batch x Embed).
            |---- y (torch.Tensor) the label associated to each image (Batch)
        OUTPUT
            |---- loss (torch.Tensor) the supervised contrastive loss for the batch.
        """
        # concat both represenation to get a (2*Batch x Embed)
        p = torch.cat((z_i, z_j), dim=0)

        # Compute the similarity matrix between elements --> (2Batch x 2Batch)
        sim = nn.CosineSimilarity(dim=2)(p.unsqueeze(1), p.unsqueeze(0)) / self.tau

        # generate mask of positive positions on the similarity matrix
        # duplicate the label tensor (size = 2N)
        y2 = torch.cat([y,y], dim=0).view(-1,1)

        if self.y_list == 'all':
            # mark as positive, position that shares the same label (1 = positive)
            mask = torch.eq(y2, y2.T).float()
        else:
            # keep only class sepcified in y_list as positive
            mask = torch.zeros((2*self.batch_size, 2*self.batch_size)).bool().to(self.device)
            for y_i in self.y_list:
                mask = mask | torch.mul((y2 == y_i), (y2 == y_i).T)

            # define representations of same image as positive
            mask = mask | torch.diag(torch.ones(self.batch_size).to(self.device), diagonal=self.batch_size).bool()
            mask = mask | torch.diag(torch.ones(self.batch_size).to(self.device), diagonal=-self.batch_size).bool()

        # remove the diagonal (self posiitve)
        mask = mask.fill_diagonal_(0).float()
        # define logit_mask : all except diagonal of similarity matrix to ignore self comparison in CE
        logit_mask = torch.ones(mask.shape).fill_diagonal_(0).to(self.device)

        # compute log_prob (i.e Cross Entropy)
        exp_logits = torch.exp(sim) * logit_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True)) # log(exp(x)/sum(exp(x_i))) = x - log(sum(exp(x_i)))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # compute loss : sum over batch
        loss = - mean_log_prob_pos.mean()

        return loss
