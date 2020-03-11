import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.utils
import sys
sys.path.append('../../..')

#from src.models.networks.ResNetBlocks import ResidualBlock#, UpResidualBlock

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
        self.bn1 = nn.BatchNorm2d(out_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channel, affine=True)
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
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, \
                               bias=False, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(in_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            # TO CHECK : Out->Out or In->Out
            self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, \
                                   bias=False, padding=1, dilation=1)
        else:
            self.conv2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, \
                                            stride=stride, bias=False, padding=1, output_padding=1)
            # TO CHECK : Out->Out or In->Out
            # self.conv2 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
            #                            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, \
            #                                      bias=False, dilation=1))

        self.bn2 = nn.BatchNorm2d(out_channel, affine=True)
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

class ResNet18_Encoder(nn.Module):
    """
    Combine multiple Residual block to form a ResNet18 up to the Average poolong
    layer. The size of the embeding dimension can be different than the one from
    ResNet18. The ResNet18 part can be initialized with pretrained weights on ImageNet.
    """
    def __init__(self, layers=[2, 2, 2, 2], pretrained=False):
        """
        Build the Encoder from the layer's specification. The encoder is composed
        of an initial 7x7 convolution that halves the input dimension (h and w)
        followed by several layers of residual blocks. Each layer is composed of
        k Residual blocks. The first one reduce the input height and width by a
        factor 2 while the number of channel is increased by 2.
        ----------
        INPUT
            |---- layer (list of int) the number of residual block to add in each
            |           layer. The length of the list represent the number of layers.
            |---- pretrained (bool) whether the ResNet18 should be loaded with
            |           pretrained weights on Imagenet.
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        block = ResidualBlock
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if pretrained: self.load_pretrain()

    def _make_layer(self, block, channel, n_blocks, stride=1):
        """
        Create a layer of residual blocks with given number of output channels.
        It also manage the downsampling to ensure that the residual can be summed
        to the convolutional output.
        ----------
        INPUT
            |---- block (nn.Module) the type of module to use as residual block.
            |---- channel (int) the output number fo channels.
            |---- n_blocks (int) the number of blocks to add in the layer.
            |---- stride (int) the stride to use in the first convolution of the layer.
        OUTPUT
            |---- layer (nn.Module) the resulting layer.
        """
        downsample = None
        # define downsample if larger stride or different channels to get similar residual
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel, affine=True)
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

class ResNet18_Decoder(nn.Module):
    """
    Combine multiple Up Residual Blocks to form a ResNet18 like decoder.
    """
    def __init__(self, layers=[2, 2, 2, 2], output_channels=3):
        """
        Build the ResNet18-like decoder. The decoder is composed of a Linear layer.
        The linear layer is interpolated (bilinear) to 512x16x16 which is then
        processed by several Up-layer of Up Residual Blocks. Each Up-layer is
        composed of k Up residual blocks. The first ones are without up sampling.
        The last one increase the input size (h and w) by a factor 2 and reduce
        the number of channels by a factor 2.
        ---------
        INPUT
            |---- layer (list of int) the number of up residual block to add in each
            |           layer. The length of the list represents the number of layers.
            |---- output_size (tuple) the decoder output size. (C x H x W)
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        block = UpResidualBlock
        self.in_channel = 512
        self.uplayer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.uplayer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.uplayer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.uplayer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.uplayer_final = nn.ConvTranspose2d(64, output_channels, kernel_size=1, stride=2, bias=False, output_padding=1)
        #self.uplayer_final = nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1)
        # self.uplayer_final = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
        #                                    nn.Conv2d(64, output_channels, kernel_size=1, stride=1, bias=False))
        self.final_activation = nn.Tanh()

    def _make_layer(self, block, out_channel, n_blocks, stride=1):
        """
        Create a layer of up residual blocks with given number of output channels.
        It also manage the upsampling to ensure that the residual can be summed
        to the convolutional output.
        ----------
        INPUT
            |---- block (nn.Module) the type of module to use as up residual block.
            |---- out_channel (int) the output number fo channels.
            |---- n_blocks (int) the number of blocks to add in the layer.
            |---- stride (int) the stride to use in the last transposed
            |           convolution of the layer.
        OUTPUT
            |---- layer (nn.Module) the resulting layer.
        """
        upsample = None
        # define upsample if larger stride or different channels at output to get similar residual
        if stride != 1 or self.in_channel != out_channel:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channel, out_channel, kernel_size=1, stride=stride, bias=False, output_padding=1),
                # nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                # nn.Conv2d(self.in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel, affine=True)
            )
        layer = []
        # first : blocks without up sampling
        for _ in range(1, n_blocks):
            layer.append(block(self.in_channel, self.in_channel))
        # last block with upsampling
        layer.append(block(self.in_channel, out_channel, stride=stride, upsample=upsample))
        self.in_channel = out_channel

        return nn.Sequential(*layer)

    def forward(self, x):
        """
        Forward pass of the decoder.
        ----------
        INPUT
            |---- x (torch.Tensor) the input with dimension (B x embed_dim).
        OUTPUT
            |---- out (torch.Tensor) the reconstructed image (B x C x H x W).
        """
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_final(x)
        x = self.final_activation(x)
        return x

class AE_SVDD_Hybrid(nn.Module):
    """
    Autoencoder based on the ResNet18. The Encoder is a ResNet18 up to the
    average pooling layer, and the decoder is a mirrored ResNet18. The embedding
    is additionnaly processed through two convolutional layers to generate a
    smaller embedding for the DeepSVDD data representation.
    """
    def __init__(self, pretrain_ResNetEnc=False, output_channels=3, return_svdd_embed=True):
        """
        Build the ResNet18 Autoencoder with the provided embeding dimension.
        The Encoder can be initialized with weights pretrained on ImageNet.
        ----------
        INPUT
            |---- pretrain_ResNetEnc (bool) whether to use pretrained weights on
            |           ImageNet for the encoder initialization.
            |---- out_channel (int) the output channel of the reconstructed image.
            |---- return_embed (bool) whether to return the DeepSVDD embedding
            |           in the forward
        OUTPUT
            |---- None
        """
        nn.Module.__init__(self)
        self.return_svdd_embed = return_svdd_embed
        self.encoder = ResNet18_Encoder(pretrained=pretrain_ResNetEnc)
        self.conv_svdd = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, bias=False),
                                       nn.BatchNorm2d(256, affine=True),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=3, stride=2),
                                       nn.Conv2d(256, 128, 3, stride=1, bias=False),
                                       nn.BatchNorm2d(128, affine=True),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, stride=1, bias=False),
                                       nn.BatchNorm2d(128, affine=True))
        # self.conv_svdd = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, bias=False),
        #                                nn.BatchNorm2d(256, affine=True),
        #                                nn.ReLU(),
        #                                nn.MaxPool2d(kernel_size=3, stride=2),
        #                                nn.Conv2d(256, 128, 5, stride=1, bias=False),
        #                                nn.BatchNorm2d(128, affine=True))

        self.decoder = ResNet18_Decoder(output_channels=output_channels)

    def forward(self, input):
        """
        Foward pass of the Autoencoder to reconstruct the provided image.
        ----------
        INPUT
            |---- input (torch.Tensor) the input image (Grayscale or RGB) with
            |           dimension (B x C x H x W).
        OUTPUT
            |---- rec (torch.Tensor) the reconstructed image (B x C' x H x W)
        """
        ae_embedding = self.encoder(input)
        rec = self.decoder(ae_embedding)
        svdd_embedding = None

        if self.return_svdd_embed:
            svdd_embedding = self.conv_svdd(ae_embedding)
            svdd_embedding = torch.flatten(svdd_embedding, start_dim=1)

        return rec, svdd_embedding

# MODIFICATION PERFORMED
# in Up residual block : keep In channel number in conv and at end of the block reduce it to Out channels number (before In-Out at the begining)
# Use of ConvTranspose2d instead of Upsampling+conv because less memory usage in forward/backward and same number of parameters
# AE embedding is 512x16x16
# SVDD CNN in the end takes 512x16x16 input and perform 3 convolution (Conv->BN->ReLU->MaxPool->Conv->BN->ReLU-Conv->BN) and provide a data embdeing in 512 dimensions.

# # %%
# import torchsummary
# m = AE_SVDD_Hybrid(pretrain_ResNetEnc=False, output_channels=1, return_svdd_embed=True)
# torchsummary.summary(m, (1, 512, 512), device='cpu', batch_size=16)
#
# with torch.no_grad():
#     input = torch.rand(4, 1, 512, 512, device='cpu', requires_grad=False)
#     rec, em = m(input)
#
# plt.imshow(input[0,0,:,:], cmap='gray')
# plt.imshow(rec[0,0,:,:], cmap='gray')

# # m = ResNet18_Encoder(pretrained=True)
# # torchsummary.summary(m, (1, 512, 512), device='cpu', batch_size=16)
# # #
# # m = ResNet18_Decoder(output_channels=1)
# # torchsummary.summary(m, (512, 16, 16), device='cpu', batch_size=16)

# m = nn.Sequential(
#         nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
#         nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
#         nn.BatchNorm2d(128, affine=True))
#
# m = nn.Sequential(
#         nn.PixelShuffle(upscale_factor=2),
#         nn.Conv2d(int(64/4), 128, kernel_size=1, stride=1, bias=False),
#         nn.BatchNorm2d(128, affine=True))
#
# m = nn.Sequential(
#         nn.ConvTranspose2d(64, 128, kernel_size=1, stride=2, bias=False, output_padding=1),
#         nn.BatchNorm2d(128, affine=True))
#
# torchsummary.summary(m, (64, 16, 16), device='cpu', batch_size=-1)

#
