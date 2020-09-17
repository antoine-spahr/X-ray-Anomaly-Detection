import torch
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    """
    Define the Supervised Contrastive Loss as a Pytorch Module.
    """
    def __init__(self, tau):
        """
        Initialize a Supervised Contrastive Loss Module.
        ----------
        INPUT
            |---- tau (float) the temperature hyperparameter.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.tau = tau

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
        # mark as positive, position that shares the same label (1 = positive)
        mask = torch.eq(y2, y2.T).float()
        # remove the diagonal (self posiitve)
        mask = mask.fill_diagonal_(0)
        # define logit_mask : all except diagonal of similarity matrix to ignore self comparison in CE
        logit_mask = torch.ones(mask.shape).fill_diagonal_(0)

        # compute log_prob (i.e Cross Entropy)
        exp_logits = torch.exp(sim) * logit_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True)) # log(exp(x)/sum(exp(x_i))) = x - log(sum(exp(x_i)))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # compute loss : sum over batch
        loss = - mean_log_prob_pos.sum()

        return loss
