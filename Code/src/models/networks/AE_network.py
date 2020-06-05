import torch
import torch.nn as nn

from src.models.networks.ResNetBlocks import DownResBlock, UpResBlock

class ResNet18_Encoder(nn.Module):
    """
    Combine multiple Residual block to form a ResNet18 up to the Average poolong
    layer. The size of the embeding dimension can be different than the one from
    ResNet18.
    """
    def __init__(self):
        """
        Build the Encoder from the layer's specification. The encoder is composed
        of an initial 7x7 convolution that halves the input dimension (h and w)
        followed by several layers of residual blocks. Each layer is composed of
        k Residual blocks. The first one reduce the input height and width by a
        factor 2 while the number of channel is increased by 2.
        ----------
        INPUT
            |---- None
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

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        """
        Forward pass of the Encoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input tensor (B x C x H x W). The input
            |           image can be grayscale or RGB. If it's grayscale it will
            |           be converted to RGB by stacking 3 copy.
        OUTPUT
            |---- out (torch.Tensor) the embedding of the image in dim 512.
        """
        # if grayscale (1 channel) convert to RGB by duplicating on 3 channel
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
        # Average pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNet18_Decoder(nn.Module):
    """
    Combine multiple Up Residual Blocks to form a ResNet18 like decoder.
    """
    def __init__(self, output_channels=3):
        """
        Build the ResNet18-like decoder. The decoder is composed of a Linear layer.
        The linear layer is interpolated (bilinear) to 512x16x16 which is then
        processed by several Up-layer of Up Residual Blocks. Each Up-layer is
        composed of k Up residual blocks. The first ones are without up sampling.
        The last one increase the input size (h and w) by a factor 2 and reduce
        the number of channels by a factor 2.
        ---------
        INPUT
            |---- output_size (tuple) the decoder output size. (C x H x W)
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)

        self.interp_layer = nn.Upsample(size=(16,16), mode='bilinear', align_corners=True)

        self.uplayer1 = nn.Sequential(UpResBlock(512, 512, upsample=False),
                                      UpResBlock(512, 256, upsample=True))
        self.uplayer2 = nn.Sequential(UpResBlock(256, 256, upsample=False),
                                      UpResBlock(256, 128, upsample=True))
        self.uplayer3 = nn.Sequential(UpResBlock(128, 128, upsample=False),
                                      UpResBlock(128, 64, upsample=True))
        self.uplayer4 = nn.Sequential(UpResBlock(64, 64, upsample=False),
                                      UpResBlock(64, 64, upsample=True))

        self.uplayer_final = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                           nn.Conv2d(64, output_channels, kernel_size=1, stride=1, bias=False))
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the decoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x embed_dim).
        OUTPUT
            |---- out (torch.Tensor) the reconstructed image (B x C x H x W).
        """
        x = x.view(-1, 512, 1, 1)
        x = self.interp_layer(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_final(x)
        x = self.final_activation(x)
        return x

class MLPHead(nn.Module):
    """

    """
    def __init__(self, Neurons_layer=[512,256,128]):
        """

        """
        nn.Module.__init__(self)
        self.fc_layers = nn.ModuleList(nn.Linear(in_features=n_in, out_features=n_out) for n_in, n_out in zip(Neurons_layer[:-1], Neurons_layer[1:]))
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        """
        for linear in self.fc_layers[:-1]:
            x = self.relu(linear(x))
        x = self.fc_layers[-1](x)
        return x

class AE_net(nn.Module):
    """

    """
    def __init__(self, MLP_Neurons_layer_enc=[512,256,128], MLP_Neurons_layer_dec=[128,256,512], output_channels=3):
        """

        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer_enc)
        self.head_dec = MLPHead(MLP_Neurons_layer_dec)
        self.decoder = ResNet18_Decoder(output_channels=output_channels)

    def forward(self, x):
        """

        """
        h = self.encoder(x)
        z = self.head_enc(h)
        # reconstruct
        rec = self.decoder(self.head_dec(z))

        return h, z, rec

class Encoder(nn.Module):
    """

    """
    def __init__(self, MLP_Neurons_layer=[512,256,128]):
        """

        """
        nn.Module.__init__(self)
        self.encoder = ResNet18_Encoder()
        self.head_enc = MLPHead(MLP_Neurons_layer)

    def forward(self, x):
        """

        """
        h = self.encoder(x)
        z = self.head_enc(h)

        return h, z
