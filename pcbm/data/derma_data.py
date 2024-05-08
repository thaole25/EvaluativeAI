import os
import torch
from PIL import Image
from .constants import DERM7_FOLDER


class Derm7ptDataset:
    def __init__(
        self,
        images,
        base_dir=os.path.join(DERM7_FOLDER, "images"),
        transform=None,
        image_key="derm",
    ):
        self.images = images
        self.transform = transform
        self.base_dir = base_dir
        self.image_key = image_key

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.images.iloc[idx]
        img_path = os.path.join(self.base_dir, row[self.image_key])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
