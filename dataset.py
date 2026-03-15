import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision

class GenDataset(Dataset):

    def __init__(self, df, img_dir, augment=None):
        self.df = df
        self.img_dir = img_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB)

        if self.augment:
            image = self.augment(image)

        label = int(self.df.iloc[idx, 1])
        return {"image": image, "label": torch.tensor(label)}