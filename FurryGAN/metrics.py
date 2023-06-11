import numpy as np
from scipy.linalg import sqrtm
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def FID(real_feats, generated_feats):
    real_feats = real_feats.numpy()
    generated_feats = generated_feats.numpy()
    # Calculate means for generated and real features
    real_mean = np.mean(real_feats, axis=0)
    generated_mean = np.mean(generated_feats, axis=0)

    # Calculate covariances for generated and real features
    real_cov = np.cov(real_feats, rowvar=False)
    generated_cov = np.cov(generated_feats, rowvar=False)

    # Calculate sum of squared difference of generated and real means
    sum_squared_diff_mean = np.sum((generated_mean - real_mean) ** 2)

    # Calculate square root of the covariance dot product
    cov_dotprod = np.dot(real_cov, generated_cov)
    sqroot_cov_dotprod = sqrtm(cov_dotprod)

    # Check if the sqroot_cov_dotprod is complex
    if np.iscomplexobj(sqroot_cov_dotprod):
        sqroot_cov_dotprod = sqroot_cov_dotprod.real

    # Calculate the FID score
    cov_trace = np.trace(real_cov + generated_cov - 2.0 * sqroot_cov_dotprod)
    fid_score = sum_squared_diff_mean + cov_trace

    return fid_score


def calculate_FID(real_imgs, generated_imgs, device):
    # Initialize an Inception v3 Model
    inception_model = inception_v3(pretrained=True)

    # To get activations set fc as identity
    inception_model.fc = torch.nn.Identity()

    # Pass the model to the device
    inception_model.to(device)
    inception_model.eval()

    # From https://pytorch.org/hub/pytorch_vision_inception_v3/
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Generate features
    with torch.no_grad():
        processed_real_imgs = preprocess(real_imgs).to(device)
        processed_generated_imgs = preprocess(generated_imgs).to(device)

        real_feats = inception_model(processed_real_imgs).detach().cpu()
        generated_feats = inception_model(processed_generated_imgs).detach().cpu()

    # Calculate the FID score
    fid_score = FID(real_feats, generated_feats)

    return fid_score


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_miou(pred_mask1, gt_mask1, pred_mask2, gt_mask2):
    iou1 = calculate_iou(pred_mask1, gt_mask1)
    iou2 = calculate_iou(pred_mask2, gt_mask2)
    return (iou1 + iou2) / 2


def calculate_recall(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    total_pixel_gt = np.sum(gt_mask)
    return intersection.sum() / total_pixel_gt


def calculate_precision(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    total_pixel_pred = np.sum(pred_mask)
    return intersection.sum() / total_pixel_pred


def calculate_f1(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    total_area = np.sum(pred_mask) + np.sum(gt_mask)
    return 2 * intersection.sum() / total_area


def calculate_accuracy(pred_mask, gt_mask):
    xnor = np.sum(pred_mask == gt_mask)
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    return xnor / (xnor + union.sum() - intersection.sum())