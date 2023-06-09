import torch
from torch import nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, in_channel, out_channel, embedding_dim):
        super().__init__()
        mid_channel = min(in_channel, out_channel)
        self.equality = in_channel != out_channel
        self.norm1 = nn.BatchNorm2d(in_channel, affine=False)
        self.conv1 = nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv_std1 = nn.Conv2d(128, in_channel, kernel_size=3, padding=1)
        self.conv_mean1 = nn.Conv2d(128, in_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)

        self.norm2 = nn.BatchNorm2d(mid_channel, affine=False)
        self.conv3 = nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv_std2 = nn.Conv2d(128, mid_channel, kernel_size=3, padding=1)
        self.conv_mean2 = nn.Conv2d(128, mid_channel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)

        if self.equality:
            self.norm3 = nn.BatchNorm2d(in_channel, affine=False)
            self.conv5 = nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1)
            self.conv_std3 = nn.Conv2d(128, in_channel, kernel_size=3, padding=1)
            self.conv_mean3 = nn.Conv2d(128, in_channel, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x, heatmaps):
        initial = x
        normalized_x = self.norm1(x)
        heatmaps_features = F.leaky_relu(self.conv1(heatmaps), 0.2)
        heatmaps_mean = self.conv_mean1(heatmaps_features)
        heatmaps_std = self.conv_std1(heatmaps_features)
        spade1 = (1+heatmaps_std) * normalized_x + heatmaps_mean
        x = self.conv2(F.leaky_relu(spade1, 0.2))

        normalized_x = self.norm2(x)
        heatmaps_features = F.leaky_relu(self.conv3(heatmaps), 0.2)
        heatmaps_mean = self.conv_mean2(heatmaps_features)
        heatmaps_std = self.conv_std2(heatmaps_features)
        spade2 = (1+heatmaps_std) * normalized_x + heatmaps_mean
        x = self.conv4(F.leaky_relu(spade2, 0.2))

        if self.equality:
            normalized_x = self.norm3(initial)
            heatmaps_features = F.leaky_relu(self.conv5(heatmaps), 0.2)
            heatmaps_std = self.conv_std3(heatmaps_features)
            heatmaps_mean = self.conv_mean3(heatmaps_features)
            spade3 = (1+heatmaps_std) * normalized_x + heatmaps_mean
            initial = self.conv6(spade3)

        return x + initial