import torch
import torch.nn as nn
import torch.nn.functional as F


class FcnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:int=3, padding=1, depth:int=2):
        super().__init__()

        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        """
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding
            ))
            self.conv_layers.append(nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.pool(x)

        return x

class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        # Block 1
        self.block1 = FcnBlock(1, 64, kernel_size=3, padding=1, depth=2)

        # Block 2
        self.block2 = FcnBlock(64, 128, kernel_size=3, padding=1, depth=2)

        # Block 3
        self.block3 = FcnBlock(128, 256, kernel_size=3, padding=1, depth=3)

        # Block 4
        self.block4 = FcnBlock(256, 512, kernel_size=3, padding=1, depth=3)

        # Block 5
        self.block5 = FcnBlock(512, 512, kernel_size=3, padding=1, depth=3)

        # -------------------------
        # Fully convolutional layers
        # -------------------------

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Classifier
        self.score = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Upsampling to go back to original size
        self.upscore = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=64,
            stride=32,
            padding=16,
            bias=False
        )

    def forward(self, x):
        input_size = x.size()

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.relu6(self.fc6(x))
        x = self.drop6(x)

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)

        x = self.score(x)

        x = self.upscore(x)

        # Crop if needed
        x = x[:, :, :input_size[2], :input_size[3]]

        return x