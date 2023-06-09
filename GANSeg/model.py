import torch
from torch import nn
import torch.nn.functional as F
from other_models import *
from utils import *
import math
import numpy as np
from spade import SPADE
from adain import AdaIN


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.latent_dim = args.latent_dim
        self.cluster_number = args.cluster_number
        self.points_per_cluster = args.n_per_kp
        self.total_points = self.points_per_cluster * self.cluster_number
        self.embedding_dim = args.embedding_dim
        self.image_size = args.image_size
        self.feature_map_sizes = [32,32,64,128]
        self.feature_map_channels = [512,256,128,64]
        self.noise_shapes = [(self.latent_dim,), (self.latent_dim,), (self.latent_dim,)]
        self.keypoints_embedding = nn.Embedding(self.cluster_number, self.embedding_dim)

        self.mask_spade_blocks = nn.ModuleList([
            SPADE(512, self.embedding_dim, self.embedding_dim),  # 32
            SPADE(self.embedding_dim, self.embedding_dim, self.embedding_dim),  # 64
            SPADE(self.embedding_dim, self.embedding_dim, self.embedding_dim),  # 128
            SPADE(self.embedding_dim, self.cluster_number + 1, self.embedding_dim),  # 128
        ])

        self.spade_blocks = nn.ModuleList([
            SPADE(512, 256, self.embedding_dim),  # 32
            SPADE(256, 128, self.embedding_dim),  # 64
            SPADE(128, self.embedding_dim, self.embedding_dim),  # 128
            SPADE(self.embedding_dim, self.embedding_dim, self.embedding_dim),  # 128
        ])

        self.adain_blocks = nn.ModuleList([
            AdaIN(512, 256, self.embedding_dim),  # 32
            AdaIN(256, 128, self.embedding_dim),  # 64
            AdaIN(128, self.embedding_dim, self.embedding_dim),  # 128
            AdaIN(self.embedding_dim, self.embedding_dim, self.embedding_dim),  # 128
        ])

        self.x_start = PositionalEmbedding(cluster_points=self.cluster_number, out_channels=self.feature_map_channels[0])
        self.mask_start = PositionalEmbedding(cluster_points=self.total_points, out_channels=self.feature_map_channels[0])
        self.bg_start = PositionalEmbedding(cluster_points=1, out_channels=self.feature_map_channels[0])

        self.rep_pad = nn.ReplicationPad2d(10)

        self.gen_keypoints_embedding_noise = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.embedding_dim)
        )

        self.gen_keypoints_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.total_points * 2)
        )

        self.gen_background_embedding = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.latent_dim, self.embedding_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.embedding_dim, 3, kernel_size=3, padding=1),
        )

        grid = grid2d_extend(self.feature_map_sizes[0], extend=10).reshape(1, -1, 2)
        self.init_extend_coord = nn.Parameter(grid, requires_grad=False)

        self.coord = {}

        for image_size in np.unique(self.feature_map_sizes):
            self.coord[str(image_size)] = grid2d(image_size).reshape(1, -1, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_dict):
        out_batch = cluster_center_generation(input_dict, self.gen_keypoints_layer, self.keypoints_embedding, self.gen_keypoints_embedding_noise, self.cluster_number, self.total_points)
        keypoints = input_dict['keypoints']
        points = input_dict['points']
        kp_emb = input_dict['kp_emb']

        diff = points.unsqueeze(-2) - self.init_extend_coord.unsqueeze(-3)  # (batch_size, total_points, self.image_sizes[0]**2, 2)
        diff = diff.transpose(2, 3).contiguous()  # (batch_size, total_points, 2, self.image_sizes[0]**2)
        diff = diff.reshape(-1, 2 * self.total_points, (self.feature_map_sizes[0] + 2 * 10), (self.feature_map_sizes[0] + 2 * 10))
        mask = self.mask_start(diff)

        tau = points.reshape(-1, self.cluster_number, self.points_per_cluster, 2) - keypoints.reshape(-1, self.cluster_number, 1, 2)
        tau = tau.norm(dim=-1).var(dim=-1, keepdim=True) + 1e-6

        for i in range(len(self.feature_map_sizes)):
            init_mask = heatmaps_extend(keypoints, self.feature_map_sizes[i], tau=tau, extend=10)
            heatmaps = init_mask.unsqueeze(2) * kp_emb.unsqueeze(-1).unsqueeze(-1)
            heatmaps = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], kp_emb.shape[2], mask.shape[-1], mask.shape[-1]).sum(dim=1)
            mask = self.mask_spade_blocks[i](mask, heatmaps)
            if i == len(self.feature_map_sizes) - 1:
                break
            elif self.feature_map_sizes[i] != self.feature_map_sizes[i + 1]:
                mask = crop(F.interpolate(mask, size=mask.shape[-1]*2, mode='bilinear', align_corners=False))

        mask = F.softmax(crop(mask), dim=1)

        input_dict['mask'] = mask
        input_dict['init_mask'] = crop(init_mask)
        keypoints = input_dict['keypoints']
        kp_emb = input_dict['kp_emb']

        diff = keypoints.unsqueeze(-2) - self.init_extend_coord.unsqueeze(-3)  # (batch_size, total_points, self.image_sizes[0]**2, 2)
        diff = diff.transpose(2, 3).contiguous()  # (batch_size, total_points, 2, self.image_sizes[0]**2)
        fg = diff.reshape(-1, 2 * self.cluster_number, (self.feature_map_sizes[0] + 2 * 10), (self.feature_map_sizes[0] + 2 * 10))
        fg = self.x_start(fg)

        for i in range(len(self.feature_map_sizes)):
            current_kp_mask = self.rep_pad(F.interpolate(mask[:, :-1, :, :], size=self.feature_map_sizes[i], mode='bilinear', align_corners=False))
            heatmaps = current_kp_mask.unsqueeze(2) * kp_emb.unsqueeze(-1).unsqueeze(-1)
            heatmaps = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], kp_emb.shape[2], fg.shape[-1], fg.shape[-1]).sum(dim=1)
            fg = self.spade_blocks[i](fg, heatmaps)

            if i == len(self.feature_map_sizes) - 1:
                break

            elif self.feature_map_sizes[i] != self.feature_map_sizes[i + 1]:
                fg = crop(F.interpolate(fg, size=fg.shape[-1]*2, mode='bilinear', align_corners=False))

        input_dict['fg'] = crop(fg)

        bg_center = input_dict['bg_trans']
        bg_center = torch.cat([torch.zeros_like(bg_center[:, :, 0:1]), bg_center[:, :, 1:2]], dim=2)
        bg_emb = self.gen_background_embedding(input_dict['input_noise1'])

        diff = bg_center.unsqueeze(-2) - self.init_extend_coord.unsqueeze(-3)  # (batch_size, 1, self.image_sizes[0]**2, 2)
        diff = diff.transpose(2, 3).contiguous()  # (batch_size, 1, 2, self.image_sizes[0]**2)
        bg = diff.reshape(-1, 2, (self.feature_map_sizes[0] + 2 * 10), (self.feature_map_sizes[0] + 2 * 10))
        bg = self.bg_start(bg)

        for i, adain_block in enumerate(self.adain_blocks):
            bg = adain_block(bg, bg_emb)
            if self.feature_map_sizes[i] != self.feature_map_sizes[min(i + 1, len(self.feature_map_sizes) - 1)]:
                bg = crop(F.interpolate(bg, size=bg.shape[-1] * 2, mode='bilinear', align_corners=False))

        input_dict['bg'] = crop(bg)
        out_batch = image_generation(input_dict, self.conv)

        out_batch['center_penalty'] = torch.tensor(0.0, device=out_batch['img'].device)
        out_batch['area_penalty'] = torch.tensor(0.0, device=out_batch['img'].device)

        return out_batch

