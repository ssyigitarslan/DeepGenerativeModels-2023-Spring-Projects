"""
   GeneratedDataset

"""

import os 
import glob
from torch.utils.data import Dataset
from torchvision.io import read_image

class GeneratedDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        super().__init__
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        data_path = os.path.join(img_dir,'*g') 
        files = glob.glob(data_path) 
  
        for file in files: 
            img = read_image(file)
            self.data.append(img / 255.) # Dividing by 255 for having range [0, 1] 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
      
        img = self.data[index]

        if self.transform:
            img = self.transform(img)

        return img
