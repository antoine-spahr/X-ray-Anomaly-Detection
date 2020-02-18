import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """

    """
    def __init__(self, reduction='mean'):
        """

        """
        nn.Module.__init__(self)
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input, target, mask):
        """

        """
        # compute loss where mask = 1
        loss = nn.criterion(input * mask, target * mask)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / torch.sum(mask)
        return loss
