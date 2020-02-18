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

    """
    def __init__(self, in_channel, out_channel, stride=1, upsample=None):
        """

        """
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, \
                                   bias=False, padding=1, dilation=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=3, \
                                            stride=stride, bias=False, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.upsample = upsample

    def forward(self, x):
        """

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

class ResNet18_Encoder(nn.Module):
    """

    """
    def __init__(self, layers=[2, 2, 2, 2], embed_dim=128, pretrained=False):
        """

        """
        nn.Module.__init__(self)
        block = ResidualBlock
        self.in_channel = 64
        self.dilation = 1
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(512, self.embed_dim, bias=False)

        if pretrained: self.load_pretrain()

    def _make_layer(self, block, channel, n_blocks, stride=1):
        """

        """
        downsample = None
        # define downsample if larger stride or different channels to get similar residual
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel)
            )
        layer = []
        # first residual block (with potential downsample)
        layer.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        self.in_channel = channel
        # next blocks without downlsampling
        for _ in range(1, n_blocks):
            layer.append(block(self.in_channel, self.in_channel))

        return nn.Sequential(*layer)

    def forward(self, x):
        """

        """
        # if grayscale conver to to RGB by duplicating on 3 channel
        if x.dim() == 3:
            x = torch.stack([x]*3, dim=1)
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
        # final average pooling and Linear layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_final(x)
        return x

    def load_pretrain(self):
        """

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

class ResNet18_Decoder(nn.Module):
    """

    """
    def __init__(self, layers=[2, 2, 2, 2], embed_dim=128, output_channel=3):#, output_size=(512, 512)):
        """

        """
        nn.Module.__init__(self)
        block = UpResidualBlock
        self.embed_dim = embed_dim
        self.fc = nn.Linear(self.embed_dim, 512)
        self.in_channel = 512
        self.uplayer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.uplayer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.uplayer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.uplayer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.uplayer_final = nn.ConvTranspose2d(64, output_channel, kernel_size=1, stride=2, bias=False, output_padding=1)

    def _make_layer(self, block, out_channel, n_blocks, stride=1):
        """

        """
        upsample = None
        # define upsample if larger stride or different channels at output to get similar residual
        if stride != 1 or self.in_channel != out_channel:
            # TO CHECK --> ConvTranspose (kernel_size of 1 with no padding or kernel_size of 3 with padding=1 ??)
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channel, out_channel, kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.BatchNorm2d(out_channel)
            )
        layer = []
        # first : blocks without down sampling
        for _ in range(1, n_blocks):
            layer.append(block(self.in_channel, self.in_channel))
        # last block with upsampling
        layer.append(block(self.in_channel, out_channel, stride=stride, upsample=upsample))
        self.in_channel = out_channel

        return nn.Sequential(*layer)

    def forward(self, x):
        """

        """
        x = self.fc(x)
        x = x.view(-1, 512, 1, 1)
        x = F.interpolate(x, size=(16,16), mode='bilinear')
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_final(x)
        return x

class AE_ResNet18(nn.Module):
    """

    """
    def __init__(self, embed_dim=128, pretrain_ResNetEnc=False, output_channel=3):
        """

        """
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.encoder = ResNet18_Encoder(embed_dim=self.embed_dim, pretrained=pretrain_ResNetEnc)
        self.decoder = ResNet18_Decoder(embed_dim=self.embed_dim, output_channel=output_channel)

    def forward(self, input):
        """

        """
        embedding = self.encoder(input)
        output = self.decoder(embedding)
        return output

# %%
from torchsummary import summary
#m = UpResidualBlock(32, 16, stride=2, upsample=nn.Sequential(nn.ConvTranspose2d(32 ,16, kernel_size=1, stride=2, bias=False, padding=0, output_padding=1), nn.BatchNorm2d(16)))
# m = ResNet18_Encoder(embed_dim=128, pretrained=True)
# summary(m, (512,512))
# m = ResNet18_Decoder(embed_dim=128, output_channel=1)
# summary(m, (1,128))
m = AE_ResNet18(embed_dim=256, pretrain_ResNetEnc=True, output_channel=1)
summary(m, (512,512))
