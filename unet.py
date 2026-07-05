import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class DoubleConv(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    """
    Defines a UNet for binary segmentation task.

    For convolutional layer:
    Output size O of a single dimension: O = (I - (K-1) + 2*P - 1)/S + 1
    with I: input size, K: kernel size, P: padding, S: stride

    For pooling layer:
    Output size O of a single dimension: O = (I + (2*P) - (K-1) -1)/S + 1

    For ConvTranspose layer:
    Output size O of a single dimension: O = (I-1)*S - 2P + (K-1) +1
    """

    def __init__(self, in_channels: int, out_channels: int, use_vgg=True):
        super().__init__()

        if use_vgg:  # initialize vgg
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            features = list(vgg.features.children())

            # ---ENCODER---
            self.enc1 = nn.Sequential(*features[:4])
            self.pool1 = features[4]

            self.enc2 = nn.Sequential(*features[5:9])
            self.pool2 = features[9]

            self.enc3 = nn.Sequential(*features[10:16])
            self.pool3 = features[16]

            self.enc4 = nn.Sequential(*features[17:23])
            self.pool4 = features[23]

            self.bottleneck = nn.Sequential(*features[24:30])
        else:
            # ---ENCODER---
            self.enc1 = DoubleConv(in_channels, 64)
            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.enc2 = DoubleConv(64, 128)
            self.pool2 = nn.MaxPool2d(kernel_size=2)

            self.enc3 = DoubleConv(128, 256)
            self.pool3 = nn.MaxPool2d(kernel_size=2)

            self.enc4 = DoubleConv(256, 512)
            self.pool4 = nn.MaxPool2d(kernel_size=2)

            self.bottleneck = DoubleConv(512, 512)

        # ---DECODER---
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Image size (C, W, H)
        # ---------------- Encoder ----------------
        x1 = self.enc1(x)  # (64, W, H)
        x = self.pool1(x1)  # (64, W/2, H/2)

        x2 = self.enc2(x)  # (128, W/2, H/2)
        x = self.pool2(x2)  # (128, W/4, H/4)

        x3 = self.enc3(x)  # (256, W/4, H/4)
        x = self.pool3(x3)  # (256, W/8, H/8)

        x4 = self.enc4(x)  # (512, W/8, W/8)
        x = self.pool4(x4)  # (512, W/16, 32)

        b = self.bottleneck(x)  # (1024, 32, 32)

        # ---------------- Decoder ----------------
        d4 = self.up4(b)  # (512, 64, 64)
        d4 = torch.cat([x4, d4], dim=1)  # (1024, 64, 64)
        d4 = self.dec4(d4)  # (512, 64, 64)

        d3 = self.up3(d4)  # (256, 128, 128)
        d3 = torch.cat([x3, d3], dim=1)  # (512, 128, 128)
        d3 = self.dec3(d3)  # (256, 128, 128)

        d2 = self.up2(d3)  # (128, 256, 256)
        d2 = torch.cat([x2, d2], dim=1)  # (256, 256, 256)
        d2 = self.dec2(d2)  # (128, 256, 256)

        d1 = self.up1(d2)  # (64, 512, 512)
        d1 = torch.cat([x1, d1], dim=1)  # (128, 512, 512)
        d1 = self.dec1(d1)  # (64, 512, 512)

        return self.outconv(d1)  # (2, 512, 512)