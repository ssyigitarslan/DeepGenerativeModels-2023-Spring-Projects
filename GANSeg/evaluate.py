import argparse
import importlib
import json
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader
import numpy as np
from utils import label2rgb, get_pt_color
import os
from model import Generator, Discriminator
from tqdm import tqdm

def evaluate(generator, test_input_batch, args, epoch):
    kp_color, pt_color = get_pt_color(args.cluster_number, args.n_per_kp)
    eval_dir = os.path.join(args.log, 'results')
    os.makedirs(eval_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        sample_batch = generator(test_input_batch)
        samples = sample_batch['img'][:64].cpu().numpy()
        keypoints = sample_batch['keypoints'][:64].cpu().numpy() * (args.image_size / 2 - 0.5) + (args.image_size / 2 - 0.5)
        kp_mask = sample_batch['kp_mask'][:64].cpu()
        bg_mask = sample_batch['bg_mask'][:64].cpu()
        color_mask = torch.cat([bg_mask, kp_mask], dim=1).max(dim=1)[1].cpu().numpy().astype(int)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)

    plt.savefig(os.path.join(eval_dir, '{}.png'.format(epoch)), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(label2rgb(color_mask[i], sample.transpose((1, 2, 0)) * 0.5 + 0.5, segment_color=kp_color, alpha=0.5))

    plt.savefig(os.path.join(eval_dir, '{}_segmaps.png'.format(epoch)), bbox_inches='tight')
    plt.close(fig)