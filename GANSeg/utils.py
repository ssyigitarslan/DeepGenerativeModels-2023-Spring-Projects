import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
from matplotlib import colors
import numpy as np

def grid2d(grid_size, device='cpu', left_end=-1, right_end=1):
    x = torch.linspace(left_end, right_end, grid_size).to(device)
    x, y = torch.meshgrid([x, x])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid


def grid2d_extend(grid_size, device='cpu', extend=10, left_end=-1, right_end=1):
    x = torch.linspace(left_end, right_end, grid_size)
    right_end = 1 + (x[1] - x[0]) * extend
    heatmap_size = grid_size + 2 * extend
    x = torch.linspace(-right_end, right_end, heatmap_size).to(device)
    x, y = torch.meshgrid([x, x])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(heatmap_size, heatmap_size, 2)
    return grid


def heatmaps_extend(input, heatmap_size=16, tau=0.01, extend=1):
    """
    :param input: (batch_size, n_points, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    """
    batch_size, n_points, _ = input.shape
    grid = grid2d_extend(heatmap_size, device=input.device, extend=extend).reshape(1, -1, 2).expand(batch_size, -1, -1)
    input_norm = (input ** 2).sum(dim=2).unsqueeze(2)
    grid_norm = (grid ** 2).sum(dim=2).unsqueeze(1)
    dist = input_norm + grid_norm - 2 * torch.bmm(input, grid.permute(0, 2, 1))
    heatmaps = torch.exp(-dist / tau)
    heatmap_size = heatmap_size + 2 * extend
    return heatmaps.reshape(batch_size, n_points, heatmap_size, heatmap_size)

def get_pt_color(n_keypoints, n_per_kp):
    if n_keypoints == 4:
        colormap = ('red', 'blue', 'yellow', 'green')
    else:

        colormap = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                    'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                    'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                    'honeydew', 'thistle')[:n_keypoints]
    pt_color = []
    kp_color = []
    for i in range(n_keypoints):
        for _ in range(n_per_kp):
            pt_color.append(colors.to_rgb(colormap[i]))
        kp_color.append(colors.to_rgb(colormap[i]))
    pt_color = np.array(pt_color)
    kp_color = np.array(kp_color)

    return kp_color, pt_color


def label2rgb(mask, image, segment_color, alpha=0.7):
    color_mask = np.zeros_like(image)
    img_size = image.shape[0]

    for i in range(segment_color.shape[0] + 1):
        if i == 0:  # bg_label
            color = np.array((0, 0, 0))
        else:
            color = segment_color[i-1]
        color_mask += (mask == i).reshape((img_size, img_size, 1)) * color.reshape((1, 1, 3))

    return (1-alpha) * image + alpha * color_mask


def penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch.autograd.grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def crop(x):
    return x[:, :, 10:-10, 10:-10]

def image_generation(input_dict, conv):
    bg = input_dict['bg']
    fg = input_dict['fg']
    mask = input_dict['mask']

    kp_mask, bg_mask = mask[:, :-1, :, :], mask[:, -1:, :, :]
    img = (1 - bg_mask) * fg + bg_mask * bg
    img = torch.tanh(conv(F.leaky_relu(img, 0.2)))

    input_dict['img'] = img
    input_dict['kp_mask'] = kp_mask
    input_dict['bg_mask'] = bg_mask

    return input_dict

def cluster_center_generation(input_dict, cluster_generation_layer, cluster_embedding_layer, cluster_noise_layer, cluster_number, total_points):
    # generate location
    z = input_dict['input_noise0']
    points = torch.tanh(cluster_generation_layer(z).reshape(-1, total_points, 2) / 20)
    keypoints = points.reshape(points.shape[0], cluster_number, -1, 2).mean(dim=2)

    # generate feature
    kp_embed_noise = input_dict['input_noise2']
    kp_fixed_emb = cluster_embedding_layer(
        torch.arange(cluster_number, device=kp_embed_noise.device).unsqueeze(0).repeat(kp_embed_noise.shape[0], 1)
    )
    kp_emb = cluster_noise_layer(kp_embed_noise)
    kp_emb = kp_fixed_emb * kp_emb.unsqueeze(1)

    input_dict['points'] = points
    input_dict['keypoints'] = keypoints
    input_dict['kp_emb'] = kp_emb

    return input_dict