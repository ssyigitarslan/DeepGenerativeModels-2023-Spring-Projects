import os
import random
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from prettytable import PrettyTable

from metrics import *


def set_fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def set_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path),
        ],
    )

    logger = logging.getLogger()
    return logger


def save_model(args, models, save_paths):
    if os.path.exists(args.model_save_path) is False:
        os.makedirs(args.model_save_path)
    for model, save_path in zip(models, save_paths):
        torch.save(model.state_dict(), save_path)


def load_model(models, load_paths):
    for model, load_path in zip(models, load_paths):
        model.load_state_dict(torch.load(load_path))


def generate_results(args, fg_generator, mask_generator, bg_generator, sample_size=2):
    with torch.no_grad():
        z_fg = torch.randn(sample_size, args.latent_size, device=args.device)
        bg_latent_size = int(args.latent_size / 4)
        z_bg = z_fg[:, :bg_latent_size]

        feature_maps, style, fg_images = fg_generator(z_fg, args.truncation)
        masks, _, _ = mask_generator(style, feature_maps, gamma=1.0)
        masks = (masks > args.mask_threshold).float()

        _, _, bg_images = bg_generator(z_bg, args.truncation)

        comp_images = fg_images * masks + bg_images * (1 - masks)
        return fg_images, masks, bg_images, comp_images


def show_image_grid(fg_images, masks, bg_images, comp_images):
    def tensor_to_array(image):
        img = image.detach().cpu().numpy()
        img = (img + 1) / 2
        img = np.transpose(img, (1, 2, 0))
        return img

    title_map = {0: "Foreground", 1: "Mask", 2: "Background", 3: "Composite"}
    n_row = fg_images.shape[0]
    fig, axs = plt.subplots(n_row, 4, figsize=(24, 12))
    for i, ax in zip(range(n_row * 4), axs.flatten()):
        row = i // 4
        col = i % 4
        if col == 0:
            img = tensor_to_array(fg_images[row])
            ax.imshow(img)
        elif col == 1:
            img = tensor_to_array(masks[row])
            ax.imshow(img, cmap='gray')
        elif col == 2:
            img = tensor_to_array(bg_images[row])
            ax.imshow(img)
        elif col == 3:
            img = tensor_to_array(comp_images[row])
            ax.imshow(img)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax.set_title(title_map[col], fontdict={'fontsize': 24})
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def mask_down_sample(original_img, new_h, new_w):
    down_sampled_mask = F.interpolate(original_img, size=(new_h, new_w), mode='bilinear')
    return down_sampled_mask


def save_generated_images(images):
    for i, img in enumerate(images):
        save_image(img, f'TRACER/data/gens/img_{i}.png')


def load_tracer_masks(batch_size=16):
    masks = []
    for i in range(batch_size):
        mask = plt.imread(f'mask/gens/img_{i}.png')
        mask = torch.tensor(mask, dtype=torch.float32)
        masks.append(mask)
    return torch.stack(masks)

############ TRAINING UTILS ############


def set_requires_gradient(model, req_grad):
    for param in model.parameters():
        param.requires_grad = req_grad


############ TESTING UTILS ############

def print_metrics(real_imgs, generated_imgs, gt_fg_masks, pred_fg_masks, 
                  gt_bg_masks, pred_bg_masks, device):
    gt_fg_masks = gt_fg_masks.detach().cpu().numpy()
    pred_fg_masks = pred_fg_masks.detach().cpu().numpy()
    gt_bg_masks = gt_bg_masks.detach().cpu().numpy()
    pred_bg_masks = pred_bg_masks.detach().cpu().numpy()

    fg_iou = np.round(calculate_iou(pred_fg_masks, gt_fg_masks), 2)
    bg_iou = np.round(calculate_iou(pred_bg_masks, gt_bg_masks), 2)
    mean_iou = np.round(calculate_miou(pred_fg_masks, gt_fg_masks, pred_bg_masks, gt_bg_masks), 2)

    recall = np.round(calculate_recall(pred_fg_masks, gt_fg_masks), 2)
    precision = np.round(calculate_precision(pred_fg_masks, gt_fg_masks), 2)
    f1_score = np.round(calculate_f1(pred_fg_masks, gt_fg_masks), 2)
    accuracy = np.round(calculate_accuracy(pred_fg_masks, gt_fg_masks), 2)

    fid = np.round(calculate_FID(real_imgs, generated_imgs, device), 2)

    # Create the Table 2
    table_2 = PrettyTable(['Dataset', 'fg IoU', 'bg IoU', 'mIoU', 'recall', 'precision', 'F1', 'Accuracy'])
    table_2.add_row(['AFHQv2-Cat (Ours)', fg_iou, bg_iou, mean_iou, recall, precision, f1_score, accuracy])
    table_2.add_row(['AFHQv2-Cat (Paper)', 0.95, 0.87, 0.91, 0.98, 0.97, 0.97, 0.95])

    # Create the Table 3
    table_3 = PrettyTable(['Dataset', 'FID'])
    table_3.add_row(['AFHQv2-Cat (Ours)', fid])
    table_3.add_row(['AFHQv2-Cat (Paper)', 6.34])

    # Print the tables
    print(table_2)
    print(table_3)
