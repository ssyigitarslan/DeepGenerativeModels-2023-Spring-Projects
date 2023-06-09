import torch
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import torch
import os
from PIL import Image
import torchvision
import h5py

def download_dataset():
    pass

class CelebAWildTrain(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild_128.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_landmark'][...]) * image_size / 128
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx].float() / 255),
                  'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]

def get_dataloader(data_root, class_name, image_size, batch_size, num_workers=0, pin_memory=True, drop_last=True):
    dataset = CelebAWildTrain(data_root, image_size)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)