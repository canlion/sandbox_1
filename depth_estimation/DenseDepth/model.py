import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class DenseDepth(nn.Module):
    """DenseDepth network"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


class Encoder(nn.Module):
    """DenseDepth encoder - DenseNet169"""

    def __init__(self):
        super().__init__()
        self.feature_extractor = models.densenet161(pretrained=True)
        self.skipped_layer = ['conv0', 'pool0', 'transition1', 'transition2', 'norm5']

    def forward(self, x):
        features = list()
        for name, layer in self.feature_extractor.features._modules.items():
            x = layer(x)
            if name in self.skipped_layer:
                features.append(x)

        return features


class Decoder(nn.Module):
    """DenseDepth decoder"""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(2208, 1664, 1, 1, 0)
        self.upconv0 = UpConv(in_ch=1664+384, out_ch=832)
        self.upconv1 = UpConv(in_ch=832+192, out_ch=416)
        self.upconv2 = UpConv(in_ch=416+96, out_ch=208)
        self.upconv3 = UpConv(in_ch=208+96, out_ch=104)
        self.conv_last = nn.Conv2d(104, 1, 3, 1, 2)

    def forward(self, features):
        x = features[-1]
        x = F.leaky_relu(self.conv0(x), .2)
        x = self.upconv0(x, features[-2])
        x = self.upconv1(x, features[-3])
        x = self.upconv2(x, features[-4])
        x = self.upconv3(x, features[-5])
        x = self.conv_last(x)
        return x


class UpConv(nn.Module):
    """UpConvolution - concat inputs and skipped inputs and bilinear up-sampling followed by 2 convolution layers."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skipped_x):
        x = F.interpolate(x, tuple(skipped_x.size()[2:]), mode='bilinear', align_corners=True)
        concat_tensor = torch.cat((x, skipped_x), 1)
        x = F.leaky_relu(self.conv0(concat_tensor), negative_slope=.2)
        x = F.leaky_relu(self.conv1(x), negative_slope=.2)
        return x
