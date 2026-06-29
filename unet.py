import torch
import torch.nn as nn

def center_crop(enc_feat: torch.Tensor, target_feat: torch.Tensor):
    """
    Crop encoder feature map to match target feature map size.
    """
    _, _, H, W = target_feat.shape
    _, _, H_enc, W_enc = enc_feat.shape
    
    delta_h = H_enc - H
    delta_w = W_enc - W
    
    top = delta_h // 2
    left = delta_w // 2

    return enc_feat.narrow(2, top, H).narrow(3, left, W)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch:int, kernel_size:int = 3, stride:int = 1, padding:int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    
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
    def __init__(self, n_class):
        super().__init__()

        # ---ENCODER---
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv(512, 1024)

        # ---DECODER---
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Image size (1, 512, 512)
        # ---------------- Encoder ----------------
        x1 = self.enc1(x) # (64, 512, 512)
        p1 = self.pool1(x1) # (64, 256, 256)

        x2 = self.enc2(p1) # (128, 256, 256)
        p2 = self.pool2(x2) # (128, 128, 128)

        x3 = self.enc3(p2) # (256, 128, 128)
        p3 = self.pool3(x3) # (256, 64, 64)

        x4 = self.enc4(p3) # (512, 64, 64)
        p4 = self.pool4(x4) # (512, 32, 32)

        b = self.bottleneck(p4) # (1024, 32, 32)

        # ---------------- Decoder ----------------
        d4 = self.up4(b) # (512, 64, 64)
        d4 = torch.cat([x4, d4], dim=1) # (1024, 64, 64)
        d4 = self.dec4(d4) # (512, 64, 64)

        d3 = self.up3(d4) # (256, 128, 128)
        d3 = torch.cat([x3, d3], dim=1) # (512, 128, 128)
        d3 = self.dec3(d3) # (256, 128, 128)

        d2 = self.up2(d3) # (128, 256, 256)
        d2 = torch.cat([x2, d2], dim=1) # (256, 256, 256)
        d2 = self.dec2(d2) # (128, 256, 256)

        d1 = self.up1(d2) # (64, 512, 512)
        d1 = torch.cat([x1, d1], dim=1) # (128, 512, 512)
        d1 = self.dec1(d1) # (64, 512, 512)

        return self.outconv(d1)  # (2, 512, 512)
    