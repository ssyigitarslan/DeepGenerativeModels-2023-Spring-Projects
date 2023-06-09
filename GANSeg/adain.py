import torch
from torch import nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, in_channel, out_channel, embedding_dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channel, affine=False)
        self.conv1 = nn.Linear(embedding_dim, 128)
        self.conv_std = nn.Linear(128, in_channel)
        self.conv_mean = nn.Linear(128, in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x, style):
        normalized_x = self.norm(x)
        style = F.leaky_relu(self.conv1(style), 0.2)
        std = self.conv_std(style).unsqueeze(-1).unsqueeze(-1)
        mean = self.conv_mean(style).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(F.leaky_relu((1+std) * normalized_x + mean, 0.2))
        return x
