import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class Encoder(nn.Module):
    """Encoder with ResNet18 or ResNet34 encoder"""
    def __init__(self, encoder, *, pretrained=False):
        super().__init__()
        self.encoder = encoder(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        if not pretrained:
            self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return [block1, block2, block3, block4, block5]


class Decoder(nn.Module):
    """Decoder for two ResNet18 or ResNet34 encoders."""
    def __init__(self, out_channels=1):
        super().__init__()
        # TODO: Initialise the layers with correct number of input and output channels
        self.up_conv6 = None  # use the up_conv() function. Number of output channels = 512
        self.conv6 = None  # use the double_conv() function. Number of output channels = 512
        self.up_conv7 = None  # use the up_conv() function. Number of output channels = 256
        self.conv7 = None  # use the double_conv() function. Number of output channels = 256
        self.up_conv8 = None  # use the up_conv() function. Number of output channels = 128
        self.conv8 = None  # use the double_conv() function. Number of output channels = 128
        self.up_conv9 = None  # use the up_conv() function. Number of output channels = 64
        self.conv9 = None  # use the double_conv() function. Number of output channels = 64
        self.up_conv10 = None  # use the up_conv() function. Number of output channels = 32
        self.conv10 = None  # Use nn.Conv2d with kernel size 1 to get the segmentation mask

        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder1_blocks, encoder2_blocks):
        # TODO: Implement the forward which concatenates the outputs of the two encoders
        #       at each stage/block to use as the input
        # TODO: Replace the 1st "1" below in torch.Size with your batch size
        assert output.shape == torch.Size([1, 1, 224, 224]), \
            f"The output shape should be same as the input image's shape but it is {output.shape} instead."


class TwoEncodersOneDecoder(nn.Module):
    def __init__(self, encoder, pretrained=True, out_channels=1):
        """
        The segmentation model to be used.
        :param encoder: resnet18 or resnet34 constructor to be used as the encoder
        :param pretrained: If True(default), the encoder will be initialised with weights
                           from the encoder trained on ImageNet
        :param out_channels: Number of output channels. The value should be 1 for binary segmentation.
        """
        super().__init__()
        # TODO: Initialise the encoders and the decoder
        self.encoder1 = None
        self.encoder2 = None
        self.decoder = None

    def forward(self, x, h_x):
        # TODO: Implement the forward pass calling the encoders and passing the outputs to the decoder
        pass
