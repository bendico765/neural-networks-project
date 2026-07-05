import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class FcnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, padding=1, depth: int = 2):
        super().__init__()

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
    def __init__(self, in_channels: int, out_channels: int, use_vgg=True):
        super().__init__()

        self.use_vgg = use_vgg
        if use_vgg:  # initialize vgg
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

            # -------------------------
            # Encoder
            # -------------------------
            self.features = vgg.features

            # -------------------------
            # Fully convolutional layers
            # -------------------------
            # Convert classifier to convolutional layers
            self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
            self.conv6.weight.data.copy_(vgg.classifier[0].weight.data.view(4096, 512, 7, 7))
            self.conv6.bias.data.copy_(vgg.classifier[0].bias.data)

            self.relu6 = nn.ReLU(inplace=True)
            self.drop6 = nn.Dropout2d()

            self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
            self.conv7.weight.data.copy_(vgg.classifier[3].weight.data.view(4096, 4096, 1, 1))
            self.conv7.bias.data.copy_(vgg.classifier[3].bias.data)

            self.relu7 = nn.ReLU(inplace=True)
            self.drop7 = nn.Dropout2d()

            self.score = nn.Conv2d(4096, out_channels, kernel_size=1)

            # Upsampling to go back to original size
            self.upscore = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=64,
                stride=32,
                padding=16,
                bias=False
            )
        else:
            # -------------------------
            # Encoder
            # -------------------------
            self.features = nn.Sequential(
                FcnBlock(in_channels, 64, kernel_size=3, padding=1, depth=2),  # Block 1
                FcnBlock(64, 128, kernel_size=3, padding=1, depth=2),  # Block 2
                FcnBlock(128, 256, kernel_size=3, padding=1, depth=3),  # Block 3
                FcnBlock(256, 512, kernel_size=3, padding=1, depth=3),  # Block 4
                FcnBlock(512, 512, kernel_size=3, padding=1, depth=3)  # Block 5
            )
            # -------------------------
            # Fully convolutional layers
            # -------------------------

            self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=3)
            self.relu6 = nn.ReLU(inplace=True)
            self.drop6 = nn.Dropout2d()

            self.conv7 = nn.Conv2d(512, 512, kernel_size=1)
            self.relu7 = nn.ReLU(inplace=True)
            self.drop7 = nn.Dropout2d()

            # Classifier
            self.score = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.size()

        x = self.features(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.score(x)

        if self.use_vgg:
            x = self.upscore(x)
        else:
            x = nn.functional.interpolate(
                x,
                size=input_size[2:],
                mode="bilinear",
                align_corners=False
            )

        # Crop if needed
        x = x[:, :, :input_size[2], :input_size[3]]

        return x