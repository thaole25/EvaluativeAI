import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from collections import defaultdict

import preprocessing.params as params

NORMALIZED_AUGMENTED_TRANS = v2.Compose(
    [
        v2.RandomResizedCrop((params.INPUT_RESIZE, params.INPUT_RESIZE)),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(params.INPUT_MEAN, params.INPUT_STD),
    ]
)

NORMALIZED_NO_AUGMENTED_TRANS = v2.Compose(
    [
        v2.Resize((params.INPUT_RESIZE, params.INPUT_RESIZE)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(params.INPUT_MEAN, params.INPUT_STD),
    ]
)

NO_NORMALIZED_AUGMENTED_TRANS = v2.Compose(
    [
        v2.RandomResizedCrop((params.INPUT_RESIZE, params.INPUT_RESIZE)),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ]
)

NO_NORMALIZED_NO_AUGMENTED_TRANS = v2.Compose(
    [
        v2.Resize((params.INPUT_RESIZE, params.INPUT_RESIZE)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ]
)

INCEPTION_TRAIN_TRANS = v2.Compose(
    [
        v2.RandomResizedCrop((350, 350)),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(params.INPUT_MEAN, params.INPUT_STD),
    ]
)

INCEPTION_VAL_TRANS = v2.Compose(
    [
        v2.Resize((350, 350)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(params.INPUT_MEAN, params.INPUT_STD),
    ]
)


class SkinCancerDataset(Dataset):
    def __init__(self, img_paths, ys, transform=None):
        self.img_paths = img_paths
        self.ys = ys
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        x = Image.open(path).convert("RGB")
        y = self.ys[index]
        if self.transform:
            x = self.transform(x)  # channel first
        return x, y, path


class BasicDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]
        return x, y


def loader_to_numpy(loader):
    X = []
    y = []
    X_paths = []
    for tx, ty, xpath in loader:
        X.append(tx.cpu().numpy())
        y.append(ty.cpu().numpy())
        X_paths.append(xpath)
    X = np.concatenate(X)
    y = np.concatenate(y)
    X_paths = np.concatenate(X_paths)
    return X, y, X_paths


def get_loader_per_class(data_dl: DataLoader):
    data_per_class = defaultdict(list)
    for batch_idx, (data, target, path) in enumerate(data_dl):
        ys = target.numpy()
        for i, y in enumerate(ys):
            data_per_class[y].append(data[i])

    loader_per_class = []
    for y in data_per_class:
        print(y, len(data_per_class[y]))
        n = len(data_per_class[y])
        ys = [y for _ in range(n)]
        assert len(ys) == len(data_per_class[y])
        loader_per_class.append(
            DataLoader(
                BasicDataset(data_per_class[y], ys),
                batch_size=params.BATCH_SIZE,
                num_workers=params.NUM_WORKERS,
                # shuffle=True
            )
        )

    return loader_per_class
