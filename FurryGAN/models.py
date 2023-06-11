import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

        # kernel = torch.tensor(
        #     [[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]
        # ).reshape(1, 1, 3, 3)
        # self.kernel = nn.Parameter(kernel, requires_grad=False)
        # self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, 1, shape[2], shape[3])
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        x = x.view(shape)
        return x


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x):
        x = self.smooth(x)
        x = F.interpolate(
            x, (x.shape[2] // 2, x.shape[3] // 2), mode="bilinear", align_corners=False
        )
        return x


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.smooth = Smooth()

    def forward(self, x):
        x = self.up_sample(x)
        x = self.smooth(x)
        return x


class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight(
            [out_channels, in_channels, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        x = F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)
        return x


class EqualizedLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_channels, in_channels])
        self.bias = nn.Parameter(torch.ones(out_channels) * bias)

    def forward(self, x):
        x = F.linear(x, self.weight(), bias=self.bias)
        return x


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, demodulate):
        super().__init__()
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight(
            [out_channels, in_channels, kernel_size, kernel_size]
        )

    def forward(self, x, s):
        b, _, h, w = x.shape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s
        if self.demodulate:
            d = torch.rsqrt((weights.pow(2)).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
            weights = weights * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_channels, *ws)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        x = x.reshape(-1, self.out_channels, h, w)
        return x


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=8):
        super().__init__()
        layers = []
        for i in range(n_mlp):
            layers.append(EqualizedLinear(latent_dim, latent_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = F.normalize(z, dim=1)
        w = self.net(z)
        return w


class ToRGB(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()
        self.to_style = EqualizedLinear(latent_dim, num_channels, bias=1.0)
        self.conv = Conv2dWeightModulate(num_channels, 3, 1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style) + self.bias.view(1, -1, 1, 1)
        x = self.activation(x)
        return x


class StyleBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels):
        super().__init__()
        self.to_style = EqualizedLinear(latent_dim, in_channels, bias=1.0)
        self.conv = Conv2dWeightModulate(in_channels, out_channels, 3, demodulate=True)
        self.scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w, noise=0):
        s = self.to_style(w)
        x = self.conv(x, s)
        x += self.scale.view(1, -1, 1, 1) * noise
        x = self.activation(x + self.bias.view(1, -1, 1, 1))
        return x, s


class GeneratorLayer(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels):
        super().__init__()
        self.style1 = StyleBlock(latent_dim, in_channels, out_channels)
        self.style2 = StyleBlock(latent_dim, out_channels, out_channels)
        self.to_rgb = ToRGB(latent_dim, out_channels)

    def forward(self, x, w, noise):
        x, _ = self.style1(x, w, noise[0])
        x, s = self.style2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, s, rgb


class Generator(nn.Module):
    def __init__(self, args, mode="foreground"):
        super().__init__()
        self.device = args.device
        image_size = args.image_size
        latent_dim = args.latent_size
        if mode == "background":
            latent_dim = int(args.latent_size / 4)
        num_channels = args.num_channels
        max_channels = args.max_channels
        feature_list = [
            min(num_channels * 2**i, max_channels)
            for i in range(int(np.log2(image_size)) - 2, -1, -1)
        ]

        self.mapping = MappingNetwork(latent_dim, n_mlp=4)
        self.initial_constant = nn.Parameter(torch.randn(1, feature_list[0], 4, 4))
        self.style = StyleBlock(latent_dim, feature_list[0], feature_list[0])
        self.to_rgb = ToRGB(latent_dim, feature_list[0])

        blocks = [
            GeneratorLayer(latent_dim, feature_list[i], feature_list[i + 1])
            for i in range(len(feature_list) - 1)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.up_sample = UpSample()
        self.activation = nn.Tanh()

    def get_noise(self, batch_size):
        noise = []
        resolution = 4
        for i in range(len(self.blocks) + 1):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(
                    batch_size, 1, resolution, resolution, device=self.device
                )
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            noise.append((n1, n2))
            resolution *= 2

        return noise

    def forward(self, z, truncation=1.0):
        if truncation < 1.0:
            z = z * truncation
        w = self.mapping(z)
        w = w[None, :, :].expand(len(self.blocks) + 1, -1, -1)
        batch_size = w.shape[1]
        noise = self.get_noise(w.shape[1])
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x, s = self.style(x, w[0], noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i, block in enumerate(self.blocks):
            x = self.up_sample(x)
            x, s, new_rgb = block(x, w[i + 1], noise[i + 1])
            rgb = self.up_sample(rgb) + new_rgb

        rgb = self.activation(rgb)
        return x, s, rgb


class MiniBatchStd(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        out = torch.cat([x, std], dim=1)
        return out


class MaskPredictor(nn.Module):
    def __init__(self, in_channels=256, out_channels=32):
        super().__init__()
        self.residual = EqualizedConv2d(in_channels, out_channels, kernel_size=1)
        self.block = nn.Sequential(
            EqualizedConv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(out_channels, out_channels, kernel_size=1),
        )
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool3d((out_channels, 1, 1))

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.relu(x + residual)
        x = self.avg_pool(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x


class DiscriminatorLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            DownSample(), EqualizedConv2d(in_channels, out_channels, kernel_size=1)
        )
        self.block = nn.Sequential(
            EqualizedConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.down_sample = DownSample()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        x = (x + residual) * self.scale
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        image_size = args.image_size
        num_channels = args.num_channels
        max_channels = args.max_channels
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, num_channels, 1), nn.LeakyReLU(0.2, True)
        )
        feature_list = [
            min(num_channels * 2**i, max_channels)
            for i in range(int(np.log2(image_size)) - 1)
        ]
        self.blocks = nn.ModuleList(
            [
                DiscriminatorLayer(feature_list[i], feature_list[i + 1])
                for i in range(len(feature_list) - 1)
            ]
        )
        self.mask_predictor = MaskPredictor(in_channels=max_channels)

        num_features = feature_list[-1] + 1
        self.std = MiniBatchStd()
        self.conv = EqualizedConv2d(num_features, num_features, kernel_size=3)
        self.linear = EqualizedLinear(2 * 2 * num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask_pred = None
        x = self.from_rgb(x)
        for i, block in enumerate(self.blocks):
            if x.shape[2] == 16:
                mask_pred = self.mask_predictor(x)
            x = block(x)
        x = self.std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        # x = self.sigmoid(x)
        return x, mask_pred


class MaskNetwork(nn.Module):
    def __init__(self, style_dim, in_channels, out_channels=32):
        super().__init__()

        self.linear = EqualizedLinear(style_dim, in_channels, bias=1.0)
        self.mod_conv = Conv2dWeightModulate(
            in_channels, out_channels, kernel_size=3, demodulate=False
        )
        self.non_linear = nn.ReLU()
        self.ch_wise_pooling = nn.AvgPool3d((style_dim, 1, 1))
        # self.ch_wise_pooling = ChannelPool(kernel_size=1)

    def forward(self, style_vec, feat_map):
        s = self.linear(style_vec)
        mask = self.mod_conv(feat_map, s)
        mask = self.non_linear(mask)
        mask = self.ch_wise_pooling(mask)
        return mask


class MaskGenerator(nn.Module):
    def __init__(self, args, in_channels=32, out_channels=32):
        super().__init__()
        style_dim = args.num_channels
        self.fine_mask_net = MaskNetwork(style_dim, in_channels, out_channels)
        self.coarse_mask_net = MaskNetwork(style_dim, in_channels, out_channels)

    def forward(self, style_vec, feat_map, gamma):
        m_fine = self.fine_mask_net(style_vec, feat_map)
        m_coarse = self.coarse_mask_net(style_vec, feat_map)

        # Min-max normalization for fine mask
        m_fine_min, m_fine_max = m_fine.min(), m_fine.max()
        m_fine = (m_fine - m_fine_min) / (m_fine_max - m_fine_min)

        # Min-max normalization for coarse mask
        m_coarse_min, m_coarse_max = m_coarse.min(), m_coarse.max()
        m_coarse = (m_coarse - m_coarse_min) / (m_coarse_max - m_coarse_min)

        # Apply clipping
        m = m_coarse + gamma * m_fine
        m = torch.clip(m, 0, 1)

        return m, m_coarse, m_fine
