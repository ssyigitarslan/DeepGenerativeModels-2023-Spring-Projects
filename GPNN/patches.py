from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided
import numpy as np

# Divide an image into overlapping PxP patches
def get_image_patches(image, patch_size):

    stride_w = image.strides[0]  # stride of width
    stride_h = image.strides[1]  # stride of height
    stride_c = image.strides[2]  # stride of channel

    width, height, _ = image.shape
    patch_width, patch_height = patch_size, patch_size

    assert width >= patch_width
    assert height >= patch_height

    num_patches_w = width - patch_width + 1
    num_patches_h = height - patch_height + 1

    patches = as_strided(image, shape=(num_patches_w, num_patches_h, patch_width, patch_height, 3),
                         strides=(stride_w, stride_h, stride_w, stride_h, stride_c))

    return patches

# Aggregate patches into a single image using a Gaussian filter
def aggregate_patches(image, output, P, sigma=0.5):

    new_image = np.zeros_like(image)
    counter = np.zeros_like(image)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = output[i, j]
            patch_weight = gaussian_filter(patch, sigma)
            new_image[i:i+P, j:j+P] += patch_weight
            counter[i:i+P, j:j+P] += 1

    new_image = new_image / counter

    return new_image