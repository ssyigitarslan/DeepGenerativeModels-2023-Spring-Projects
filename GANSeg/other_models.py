import torch
from torch import nn
import torch.nn.functional as F
import math

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
        )
    
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_dict):
        x = input_dict['img']
        out = self.net(x)
        return self.fc(out.view(out.shape[0], -1)).squeeze()





class PositionalEmbedding(nn.Module):
    def __init__(self, cluster_points, out_channels):
        super().__init__()
        self.pe = nn.Conv2d(2*cluster_points, out_channels // 2, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pe(x) * math.pi
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=1)
        return x

class Embedding(nn.Module):
    def __init__(self, number, latent_dim, embedding_dim):
        super().__init__()
        self.layers = []
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        for i in range(number):
            self.layers.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers.append(nn.Linear(self.latent_dim, self.embedding_dim))
        self.embedding = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.embedding(x)
