import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.utils

class ResidualBlock(nn.Module):
    """
    Residual Block for the ResNet18 (without bottleneck).
    """
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
         in ->[Conv3x3]->[BN]->[ReLU]->[Conv3x3]->[BN]-> + -> out
            |_______________[downsample]_________________|
        ----------
        INPUT
            |---- in_channel (int) the number of input channels.
            |---- out_channel (int) the number of output channels.
            |---- stride (int) the stride for the first 3x3 convolution. Larger
            |           than one produces a size reduction of the input.
            |---- downsample (nn.Module) the downsampling module to use in order
            |           to get similar shaped residuals.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, \
                               bias=False, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass of the Residual Block.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W) with C = in_channel
        OUTPUT
            |---- out (torch.Tensor) the output tensor (B x C x H' x W') with
            |           C = out_channel. H and W are changed if the stride is
            |           bigger than one.
        """
        identity = x
        # convolution n°1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # convolution n°2
        out = self.conv2(out)
        out = self.bn2(out)
        # modify down sample if provided
        if self.downsample is not None:
            identity = self.downsample(x)
        # sum convolution with shortcut
        out += identity
        out = self.relu(out)
        return out

class UpResidualBlock(nn.Module):
    """
    Up Residual Block for the ResNet18-like decoder (without bottleneck).
    """
    def __init__(self, in_channel, out_channel, stride=1, upsample=None):
        """
         in ->[Conv3x3]->[BN]->[ReLU]->[Conv3x3 / ConvTransp3x3]->[BN]-> + -> out
            |__________________________[upsample]________________________|
        ----------
        INPUT
            |---- in_channel (int) the number of input channels.
            |---- out_channel (int) the number of output channels.
            |---- stride (int) the stride for the first 3x3 convolution. Larger
            |           than one produces a size augmentation of the input.
            |---- downsample (nn.Module) the downsampling module to use in order
            |           to get similar shaped residuals.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        # TO CHECK : In->Out or In->In
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            # TO CHECK : Out->Out or In->Out
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, \
                                   bias=False, padding=1, dilation=1)
        else:
            # self.conv2 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=3, \
            #                                 stride=stride, bias=False, padding=1, output_padding=1)
            # TO CHECK : Out->Out or In->Out
            self.conv2 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                       nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, \
                                                 bias=False, dilation=1))

        self.bn2 = nn.BatchNorm2d(out_channel)
        self.upsample = upsample

    def forward(self, x):
        """
        Forward pass of the UP Residual Block.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W) with C = in_channel
        OUTPUT
            |---- out (torch.Tensor) the output tensor (B x C x H' x W') with
            |           C = out_channel. H and W are changed if the stride is
            |           bigger than one.
        """
        identity = x
        # convolution n°1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # convolution n°2 or transposed convolution
        out = self.conv2(out)
        out = self.bn2(out)
        # modify down sample if provided
        if self.upsample is not None:
            identity = self.upsample(x)
        # sum convolution with shortcut
        out += identity
        out = self.relu(out)
        return out
