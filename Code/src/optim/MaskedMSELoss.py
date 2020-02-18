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
            |---- reduction (str) the reduction to use on the loss. ONLY mean supported so far.
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
        loss = nn.criterion(input * mask, target * mask)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / torch.sum(mask)
        return loss
