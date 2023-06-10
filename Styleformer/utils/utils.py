"""

UTILS.PY consists of the following classe definitions:

1. EqualizedWeight
2. EqualizedConv2d
3. EqualizedLinear
4. Conv2dWeightModulate
5. ToRGBLayer
6. Smooth
7. DownSample
8. UpSample

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
1. EqualizedWeight

**Disclaimer:** This class is taken from LABML's simplified StyleGAN2 implementation.
"""

class EqualizedWeight(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = torch.nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c


"""
2. EqualizedConv2d

**Disclaimer:** This class is taken from LABML's simplified StyleGAN2 implementation.
"""

class EqualizedConv2d(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):

        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


"""
3. EqualizedLinear

**Disclaimer:** This class is taken from LABML's simplified StyleGAN2 implementation. See [their description](https://nn.labml.ai/gan/stylegan/index.html#equalized_weights) on what it does.
"""

class EqualizedLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):

        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = torch.nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


"""
4. Conv2dWeightModulate

**Disclaimer**: This class is taken from LABML's StyleGAN2 implementation. See [their annotated code](https://nn.labml.ai/gan/stylegan/index.html#section-64) for details.
"""

class Conv2dWeightModulate(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):

        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)


"""
5. ToRGBLayer

**Disclaimer**: This class is mostly based on LABML's StyleGAN2 implementation.  See [their annotated code](https://nn.labml.ai/gan/stylegan/index.html#to_rgb) for details.
"""

class ToRGBLayer(torch.nn.Module):
    #def __init__(self, d_latent: int, features: int):
    def __init__(self, in_channels, out_channels, h_dim_w):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        # self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.to_style = EqualizedLinear(h_dim_w, in_channels, bias=1.0)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels).uniform_(-1./math.sqrt(in_channels), 1./math.sqrt(in_channels)))

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size=1, demodulate=False)
        
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):

        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution

        # NOTE: Styleformer states that
        # "Originally ToRGB layer converts high-dimensional per pixel data 
        # into RGB per pixel data via 1×1 convolution operation, 
        # which we replace it to same operation, linear operation
        # But they don't give further details. 
        # So we kept the ToRGB of StyleGAN2 as is.
        x = self.conv(x, style)

        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


"""
6. Smooth

**Disclaimer:** This class is based on LABML's StyleGAN2 implementation. See [their annotated code](https://nn.labml.ai/gan/stylegan/index.html#smooth) for details.
"""

class Smooth(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = torch.nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = torch.nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
       
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.reshape(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)



"""
7. DownSample

**Disclaimer:** This class is based on LABML's StyleGAN2 implementation. See [their annotated code](https://nn.labml.ai/gan/stylegan/index.html#down_sample) for details.
"""

class DownSample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Smoothing or blurring
        x = self.smooth(x)
        # Scaled down
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)

"""
8. UpSample

**Disclaimer:** This class is based on LABML's StyleGAN2 implementation. See [their annotated code](https://nn.labml.ai/gan/stylegan/index.html#up_sample) for details.
"""

class UpSample(torch.nn.Module):
  
    def __init__(self):
        super().__init__()
        # Up-sampling layer
        self.up_sample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        x = x.to(dtype=torch.float32)
        return self.smooth(self.up_sample(x))
