"""
METRICS.PY

"""

import torch
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy import linalg
from scipy.stats import entropy
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.inception import inception_v3


# --------------------------------------------------------------------------------------------------------------------
# FID Score utilities
# --------------------------------------------------------------------------------------------------------------------


class InceptionV3(nn.Module):
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()
        
        inception = torchvision.models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """
        Get Inception activations
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

"""
Get activations from given images and inceptionv3
""" 
def _get_activation(
    images,         # ### TODO ###
    model,          # ### TODO ###
    device,         # ### TODO ###
    dims=2048       # ### TODO ###
    ):

    model.eval()
    act=np.empty((len(images), dims))
    
    if(type(images) is np.ndarray):
      images = torch.from_numpy(images)

    batch=images.to(device)
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    return act 


"""
Calculates the frechet distance
"""
def _calculate_fid(
    mu1,            # ### TODO ###
    sigma1,         # ### TODO ###
    mu2,            # ### TODO ###
    sigma2,         # ### TODO ###
    eps = 1e-6      # ### TODO ###
    ):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def get_fid_score(configs, real_dataloader, gen_dataloader):

    # Load Inception V3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model = model.to(configs.device)

    # Compute FID score over real and fake images
    activation_real = []
    activation_fake = []
    for i,(imgs, generated_imgs) in enumerate(zip(real_dataloader ,gen_dataloader)):
        # Note that images have range [0,1]
        # Get the activations from Inception V3
        act_real = _get_activation(imgs, model, configs.device)
        act_fake = _get_activation(generated_imgs, model, configs.device)

        activation_real.append(act_real)
        activation_fake.append(act_fake)
    
    # To get the activation statistics (mean and std),
    # merge the activations of minibatches 
    # and compute the mean and variance over the whole dataset
    act = np.vstack(activation_real)
    act2 = np.vstack(activation_fake)

    mu_1 = np.mean(act, axis=0)
    std_1 = np.cov(act, rowvar=False)
    mu_2 = np.mean(act2, axis=0)
    std_2 = np.cov(act2, rowvar=False)

    # Get Fretched Distance
    fid_score = _calculate_fid(mu_1, std_1, mu_2, std_2)

    return fid_score

# --------------------------------------------------------------------------------------------------------------------
# Inception Score (IS)
# --------------------------------------------------------------------------------------------------------------------

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
        Computes the inception score of the generated images

        imgs       : Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda       : whether or not to run on GPU
        batch_size : batch size for feeding into Inception v3
        splits     : number of splits
    """     
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(tqdm(dataloader, desc = "Getting predictions")):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in tqdm(range(splits), desc = "IS calculation"):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
