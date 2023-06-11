import numpy as np
import torch
import torch.nn.functional as F

from utils import mask_down_sample


def d_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def r1_loss(real_pred, real_img):
    (grad_real,) = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def mask_prediction_loss(m_true, m_pred):
    norm = torch.linalg.norm(m_true)
    down_sampled_mask = mask_down_sample(m_true, 16, 16)
    loss = torch.sum(torch.square(down_sampled_mask - m_pred)) / norm
    return loss


def mask_consistency_loss(m_comp, m_fg):
    norm = torch.norm(m_comp)
    loss = torch.sum(torch.square(m_fg - m_comp)) / norm
    return loss


def coarse_mask_loss(m_coarse, phi1, device):
    binary_loss = torch.mean(torch.min(m_coarse, 1 - m_coarse))
    b, c, h, w = m_coarse.shape
    num_pixel = h * w
    area_loss = torch.mean(
        torch.max(
            torch.zeros(b).to(device), phi1 - torch.sum(m_coarse.reshape(b, -1), dim=1) / num_pixel
        )
    )
    return binary_loss + area_loss


def fine_mask_loss(m_fine, phi2, device):
    b, c, h, w = m_fine.shape
    num_pixel = h * w
    loss = torch.mean(
        torch.max(
            torch.zeros(b).to(device), phi2 - torch.sum((1 - m_fine).reshape(b, -1), dim=1) / num_pixel
        )
    )
    return loss


def background_loss(x_comp, x_bg):
    norm = torch.norm(x_comp)
    loss = torch.sum(torch.square(x_comp - x_bg)) / norm
    return loss
