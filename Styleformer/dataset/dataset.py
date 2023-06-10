"""
Dataset
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# Overriding Pytorch's CIFAR10 Dataset to remove labels
class CIFAR10(torchvision.datasets.CIFAR10):

    def __getitem__(self, index: int):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img


class STL10(torchvision.datasets.STL10):

    def __getitem__(self, index: int):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img
