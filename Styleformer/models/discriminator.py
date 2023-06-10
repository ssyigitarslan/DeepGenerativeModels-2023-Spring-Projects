"""

DISCRIMINATOR.PY consists of the following class definitions:

1.   Discriminator
1.1. DiscriminatorBlock
1.2. MiniBatchStdDev

"""

import math
import numpy as np
import torch
import torch.nn as nn

from utils import EqualizedConv2d, EqualizedLinear, DownSample

"""
1. Discriminator

**Disclaimer:** This class is based on [StyleGAN2-ADA's Discriminator](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py) as well as [LABML's StyleGAN2 Discriminator](https://nn.labml.ai/gan/stylegan/index.html#discriminator_block).
"""

class Discriminator(torch.nn.Module):
    def __init__(self,
        im_resolution,                 # Input resolution.
        channel_base,                   # Overall multiplier for the number of channels.
        channel_max,                    # Maximum number of channels in any layer.
        last_resolution,                # Resolution of the last layer
        mbstd_group_size,               # Group size for the minibatch standard deviation layer
    ):
        super().__init__()
        self.im_resolution = im_resolution
        self.im_resolution_log2 = int(np.log2(im_resolution))
        self.img_channels = 3
        self.block_resolutions = [2 ** i for i in range(self.im_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        

        channels_list = list(channels_dict.values())
        
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(in_features=self.img_channels, out_features=channels_list[0], kernel_size=1),
            nn.LeakyReLU(0.2, True),
            )
        
        n_blocks = len(channels_list) - 1
        blocks = [DiscriminatorBlock(channels_list[i], channels_list[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

         # [Mini-batch Standard Deviation](#mini_batch_std_dev)
        self.std_dev = MiniBatchStdDev(group_size=mbstd_group_size)

        final_features = channels_list[-1] + 1

        self.conv = EqualizedConv2d(
            in_features=final_features, out_features=final_features, kernel_size=3
            )

        # Final linear layer to get the classification
        self.final = EqualizedLinear(2 * 2 * final_features, 1)


    def forward(self, img):
        """
        * `img` is the input image of shape `[batch_size, 3, height, width]`
        """
        # Try to normalize the image (this is totally optional, but sped up the early training a little)
        img = img - 0.5
        # Convert from RGB
        img = self.from_rgb(img)
        # Run through the [discriminator blocks](#discriminator_block)
        img = self.blocks(img)
        # Calculate and append [mini-batch standard deviation](#mini_batch_std_dev)
        img = self.std_dev(img)
        # $3 \times 3$ convolution
        img = self.conv(img)
        # Flatten
        img = img.reshape(img.shape[0], -1)
        # Return the classification score
        return self.final(img)


"""
1.1. DiscriminatorBlock

**Disclaimer:** This class is based on [StyleGAN2-ADA's Discriminator](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py) as well as [LABML's StyleGAN2 Discriminator](https://nn.labml.ai/gan/stylegan/index.html#discriminator_block).
"""

class DiscriminatorBlock(torch.nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()
        # Down-sampling and $1 \times 1$ convolution layer for the residual connection
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        # Two $3 \times 3$ convolutions
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2, True),
        )

        # Down-sampling layer
        self.down_sample = DownSample()

        # Scaling factor $\frac{1}{\sqrt 2}$ after adding the residual
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        # Get the residual connection
        residual = self.residual(x)

        # Convolutions
        x = self.block(x)
        # Down-sample
        x = self.down_sample(x)

        # Add the residual and scale
        return (x + residual) * self.scale


"""
1.2. MiniBatchStdDev

**Disclaimer:** This class is based on [LABML's StyleGAN2](https://nn.labml.ai/gan/stylegan/index.html#discriminator_block).
"""

class MiniBatchStdDev(nn.Module):

    def __init__(self, group_size: int):
        """
        * `group_size` is the number of samples to calculate standard deviation across.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        * `x` is the feature map
        """
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        # Split the samples into groups of `group_size`, we flatten the feature map to a single dimension
        # since we want to calculate the standard deviation for each feature.
        grouped = x.view(self.group_size, -1)
        # Calculate the standard deviation for each feature among `group_size` samples
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)
