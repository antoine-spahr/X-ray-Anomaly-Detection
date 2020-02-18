import torch
import torch.nn as nn

class DeepSADLoss(nn.Module):
    """
    Implementation of the DeepSAD loss proposed by Lukas Ruff et al. (2019)
    """
    def __init__(self, c, eta, eps=0.1):
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
        self.c = c
        self.eta = eta
        self.eps = 0.1

    def forward(input, semi_target):
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
        dist = torch.sum((self.c - input)**2, dim=1)
        # compute the loss depening on the semi-supervized label
        # keep distance if semi-label is 0 or 1 (normal sample or unknonw (assumed) normal)
        # inverse distance if semi-label = -1 (known abnormal)
        losses = torch.where(semi_target == 0, dist, self.eta * ((dist + self.eps) ** semi_target.float()))
        loss = torch.mean(losses)
        return loss
