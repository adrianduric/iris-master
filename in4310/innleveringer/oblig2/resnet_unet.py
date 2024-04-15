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

        self.up_conv6 = up_conv(in_channels=1024, out_channels=512)
        self.conv6 = double_conv(in_channels=1024, out_channels=512)

        self.up_conv7 = up_conv(in_channels=512, out_channels=256)
        self.conv7 = double_conv(in_channels=512, out_channels=256)

        self.up_conv8 = up_conv(in_channels=256, out_channels=128)
        self.conv8 = double_conv(in_channels=256, out_channels=128)

        self.up_conv9 = up_conv(in_channels=128, out_channels=64)
        self.conv9 = double_conv(in_channels=192, out_channels=64)

        self.up_conv10 = up_conv(in_channels=64, out_channels=32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder1_blocks, encoder2_blocks):
        x = torch.cat((encoder1_blocks[0], encoder2_blocks[0]), dim=1)
        x = self.up_conv6(x)

        x = torch.cat((x, encoder1_blocks[1], encoder2_blocks[1]), dim=1)
        x = self.conv6(x)
        x = self.up_conv7(x)

        x = torch.cat((x, encoder1_blocks[2], encoder2_blocks[2]), dim=1)
        x = self.conv7(x)
        x = self.up_conv8(x)

        x = torch.cat((x, encoder1_blocks[3], encoder2_blocks[3]), dim=1)
        x = self.conv8(x)
        x = self.up_conv9(x)

        x = torch.cat((x, encoder1_blocks[4], encoder2_blocks[4]), dim=1)
        x = self.conv9(x)
        x = self.up_conv10(x)
        
        x = self.conv10(x)

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
        self.encoder1 = Encoder(encoder=encoder, pretrained=pretrained)
        self.encoder2 = Encoder(encoder=encoder, pretrained=pretrained)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, x, h_x):
        # TODO: Implement the forward pass calling the encoders and passing the outputs to the decoder
        encoder1_blocks = self.encoder1(x)
        encoder2_blocks = self.encoder2(h_x)
        seg_mask = self.decoder(encoder1_blocks, encoder2_blocks)

# Task 1a)
# Finding number of channels in output blocks of encoder:
if __name__ == '__main__':
    import os
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet18

    root_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = "data/train/tiled_images"
    img_filename = "train_1_0.png"
    img_path = os.path.join(root_path, images_dir, img_filename)
    img = Image.open(img_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)

    if torch.cuda.is_available():
        img = img.to('cuda')

    encoder = Encoder(encoder=resnet18)
    encoder.to('cuda')

    with torch.no_grad():
        output = encoder(img)
    block1 = output[0]
    block2 = output[1]
    block3 = output[2]
    block4 = output[3]
    block5 = output[4]
    print(f"block1.shape: {block1.shape}") # torch.Size([1, 64, 112, 112]) (64 channels)
    print(f"block2.shape: {block2.shape}") # torch.Size([1, 64, 56, 56]) (64 channels)
    print(f"block3.shape: {block3.shape}") # torch.Size([1, 128, 28, 28]) (128 channels)
    print(f"block4.shape: {block4.shape}") # torch.Size([1, 256, 14, 14]) (256 channels)
    print(f"block5.shape: {block5.shape}") # torch.Size([1, 512, 7, 7]) (512 channels)
