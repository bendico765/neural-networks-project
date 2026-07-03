import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ConvReLU(nn.Module):
    """
    Module per encoder and decoder parts of the network, which performs the operations
    of convolution, batch normalization and relu
    """

    def __init__(
            self,
            in_c,
            out_c,
            kernel_size: int = 3,
            padding: int = 1,
            conv_weight_data=None,
            conv_bias_data=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        if conv_weight_data is not None:
            self.conv.weight.data.copy_(conv_weight_data)
        if conv_bias_data is not None:
            self.conv.bias.data.copy_(conv_bias_data)

        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, depth: int = 2, kernel_size: int = 3, padding: int = 1, logits: bool = False):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.layers = nn.ModuleList()
        for i in range(depth - 1):
            self.layers.append(ConvReLU(in_c, in_c, kernel_size=kernel_size, padding=padding))

        if logits:
            self.layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding))
        else:
            self.layers.append(ConvReLU(in_c, out_c, kernel_size=kernel_size, padding=padding))

    def forward(self, x, ind):
        x = self.unpool(x, ind)
        for layer in self.layers:
            x = layer(x)
        return x


class SegNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: int = 64, use_vgg = True):
        super(SegNet, self).__init__()

        # initialize vgg
        if use_vgg:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            conv_features = [layer for layer in vgg.features if isinstance(layer, nn.Conv2d)][:13]

        ############ ENCODER
        # BLOCK 1
        self.convrelu1 = ConvReLU(
            in_channels,
            features,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[0].weight.data if use_vgg else None,
            conv_bias_data=conv_features[0].bias.data if use_vgg else None
        )

        self.convrelu2 = ConvReLU(
            features,
            features,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[1].weight.data if use_vgg else None,
            conv_bias_data=conv_features[1].bias.data if use_vgg else None
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # BLOCK 2
        self.convrelu3 = ConvReLU(
            features,
            features * 2,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[2].weight.data if use_vgg else None,
            conv_bias_data=conv_features[2].bias.data if use_vgg else None
        )

        self.convrelu4 = ConvReLU(
            features * 2,
            features * 2,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[3].weight.data if use_vgg else None,
            conv_bias_data=conv_features[3].bias.data if use_vgg else None
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # BLOCK 3
        self.convrelu5 = ConvReLU(
            features * 2,
            features * 4,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[4].weight.data if use_vgg else None,
            conv_bias_data=conv_features[4].bias.data if use_vgg else None
        )

        self.convrelu6 = ConvReLU(
            features * 4,
            features * 4,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[5].weight.data if use_vgg else None,
            conv_bias_data=conv_features[5].bias.data if use_vgg else None
        )

        self.convrelu7 = ConvReLU(
            features * 4,
            features * 4,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[6].weight.data if use_vgg else None,
            conv_bias_data=conv_features[6].bias.data if use_vgg else None
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # BLOCK 4
        self.convrelu8 = ConvReLU(
            features * 4,
            features * 8,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[7].weight.data if use_vgg else None,
            conv_bias_data=conv_features[7].bias.data if use_vgg else None
        )

        self.convrelu9 = ConvReLU(
            features * 8,
            features * 8,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[8].weight.data if use_vgg else None,
            conv_bias_data=conv_features[8].bias.data if use_vgg else None
        )

        self.convrelu10 = ConvReLU(
            features * 8,
            features * 8,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[9].weight.data if use_vgg else None,
            conv_bias_data=conv_features[9].bias.data if use_vgg else None
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        ############ BOTTLENECK
        self.convrelu11 = ConvReLU(
            features * 8,
            features * 8,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[10].weight.data if use_vgg else None,
            conv_bias_data=conv_features[10].bias.data if use_vgg else None
        )

        self.convrelu12 = ConvReLU(
            features * 8,
            features * 8,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[11].weight.data if use_vgg else None,
            conv_bias_data=conv_features[11].bias.data if use_vgg else None
        )

        self.convrelu13 = ConvReLU(
            features * 8,
            features * 8,
            kernel_size=3,
            padding=1,
            conv_weight_data=conv_features[12].weight.data if use_vgg else None,
            conv_bias_data=conv_features[12].bias.data if use_vgg else None
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck_dec = DecoderBlock(features * 8, features * 8, depth=3)

        ############ DECODER
        self.dec0 = DecoderBlock(features * 8, features * 4, depth=3)
        self.dec1 = DecoderBlock(features * 4, features * 2, depth=3)
        self.dec2 = DecoderBlock(features * 2, features)
        self.dec3 = DecoderBlock(features, out_channels, logits=True)  # No activation

    def forward(self, x):
        # encoder
        cr1 = self.convrelu1(x)
        cr2 = self.convrelu2(cr1)
        e0, ind0 = self.pool1(cr2)

        cr3 = self.convrelu3(e0)
        cr4 = self.convrelu4(cr3)
        e1, ind1 = self.pool2(cr4)

        cr5 = self.convrelu5(e1)
        cr6 = self.convrelu6(cr5)
        cr7 = self.convrelu7(cr6)
        e2, ind2 = self.pool3(cr7)

        cr8 = self.convrelu8(e2)
        cr9 = self.convrelu9(cr8)
        cr10 = self.convrelu10(cr9)
        e3, ind3 = self.pool4(cr10)

        # bottleneck
        cr11 = self.convrelu11(e3)
        cr12 = self.convrelu12(cr11)
        cr13 = self.convrelu13(cr12)
        b0, indb = self.pool5(cr13)

        b1 = self.bottleneck_dec(b0, indb)

        # decoder
        d0 = self.dec0(b1, ind3)
        d1 = self.dec1(d0, ind2)
        d2 = self.dec2(d1, ind1)

        # classification layer
        output = self.dec3(d2, ind0)
        return output