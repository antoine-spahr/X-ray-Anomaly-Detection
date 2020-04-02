import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.utils

from src.models.networks.ResNetBlocks import DownResBlock, UpResBlock

class ResNet18_binary(nn.Module):
    """

    """
    def __init__(self, pretrained=False):
        """
        Build ResNet18 binary classifer. The encoder is composed
        of an initial 7x7 convolution that halves the input dimension (h and w)
        followed by several layers of residual blocks. Each layer is composed of
        k Residual blocks. The first one reduce the input height and width by a
        factor 2 while the number of channel is increased by 2. The the output
        of the convolutional layers is .........
        ----------
        INPUT
            |---- pretrained (bool) whether the ResNet18 should be loaded with
            |           pretrained weights on Imagenet.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        # First convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual layers
        self.layer1 = nn.Sequential(DownResBlock(64, 64, downsample=False),
                                    DownResBlock(64, 64, downsample=False))
        self.layer2 = nn.Sequential(DownResBlock(64, 128, downsample=True),
                                    DownResBlock(128, 128, downsample=False))
        self.layer3 = nn.Sequential(DownResBlock(128, 256, downsample=True),
                                    DownResBlock(256, 256, downsample=False))
        self.layer4 = nn.Sequential(DownResBlock(256, 512, downsample=True),
                                    DownResBlock(512, 512, downsample=False))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_classifier = nn.Linear(512, 1, bias=False)
        #self.out_softmax = nn.Softmax(dim=1)

        if pretrained: self.load_pretrain()

    def forward(self, x):
        """
        Forward pass of the Encoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W). The input
            |           image can be grayscale or RGB. If it's grayscale it will
            |           be converted to RGB by stacking 3 copy.
        OUTPUT
            |---- out (torch.Tensor) the embedding of the image x in embed_dim
            |           vector dimension.
        """
        # if grayscale (1 channel) convert to to RGB by duplicating on 3 channel
        # assuming shape : (... x C x H x W)
        if x.shape[-3] == 1:
            x = torch.cat([x]*3, dim=1)
        # first 1x1 convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 4 layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # classification
        x = self.avg_pool(x)
        x = self.fc_classifier(x.view(x.size(0), -1))

        return x

    def load_pretrain(self):
        """
        Initialize the Encoder's weights with the weights pretrained on ImageNet.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        # download ResNet18 trained on ImageNet state dict
        pretrainResNet18_state_dict = torchvision.models.utils.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        # Get the modified ResNet Encoder state dict
        model_state_dict = self.state_dict()
        # keep only matching keys
        pretrained_dict = {k: v for k, v in pretrainResNet18_state_dict.items() if k in model_state_dict}
        # upadte state dict
        model_state_dict.update(pretrained_dict)
        self.load_state_dict(model_state_dict)
