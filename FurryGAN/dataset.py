import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DataGenerator(Dataset):
    def __init__(self, image_directory_path, mask_directory_path, args):
        self.image_path = image_directory_path
        self.mask_path = mask_directory_path
        self.image_annotations = pd.DataFrame(os.listdir(image_directory_path))
        self.mask_annotations = pd.DataFrame(os.listdir(mask_directory_path))
        self.args = args

    def __getitem__(self, item):
        image_name = os.path.join(self.image_path, self.image_annotations.iloc[item, 0])
        image = Image.open(image_name).convert("RGB")
        mask_name = os.path.join(self.mask_path, self.mask_annotations.iloc[item, 0])
        mask = Image.open(mask_name).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize([self.args.image_size, self.args.image_size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        mask = transform(mask)
        image = transform(image)

        return image, mask

    def __len__(self):
        return len(self.image_annotations)


def load_data(args):
    train_set = DataGenerator(
        image_directory_path=os.path.join(args.data_path, "object/train"),
        mask_directory_path=os.path.join(args.data_path, "mask/train"),
        args=args,
    )
    test_set = DataGenerator(
        image_directory_path=os.path.join(args.data_path, "object/test"),
        mask_directory_path=os.path.join(args.data_path, "mask/test"),
        args=args,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    return train_loader, test_loader
