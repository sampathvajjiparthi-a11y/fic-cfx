"""Dataset loading.

Supervisor note: use one SMALL and one BIG dataset.
  * small: Fashion-MNIST (Xiao, Rasul and Vollgraf, 2017)
  * big:   CIFAR-10 (approved form) or any fashion dataset stored as
           data/raw/<folder>/<class>/*.jpg (e.g. Fashion Product Images)

All images are returned as float tensors in [0, 1] with shape (C, H, W).
No normalisation is applied here, so counterfactual pixel changes stay
easy to read. Models normalise inputs internally if they need to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from fic.config import Config, data_path

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class DatasetBundle:
    name: str
    train: Dataset
    val: Dataset
    test: Dataset
    classes: list[str]
    in_channels: int
    image_size: int
    extra: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def _split(train_full: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(train_full, [n_train, n_val], generator=gen)


def _fashion_mnist(root: Path, cfg: Config) -> DatasetBundle:
    tf = transforms.ToTensor()
    train_full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    train, val = _split(train_full, cfg.datasets.val_fraction, cfg.project.split_seed)
    return DatasetBundle("fashion_mnist", train, val, test, FASHION_CLASSES, 1, 28)


def _cifar10(root: Path, cfg: Config) -> DatasetBundle:
    tf = transforms.ToTensor()
    train_full = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    train, val = _split(train_full, cfg.datasets.val_fraction, cfg.project.split_seed)
    return DatasetBundle("cifar10", train, val, test, CIFAR10_CLASSES, 3, 32)


def _image_folder(root: Path, cfg: Config) -> DatasetBundle:
    """Any fashion dataset arranged as <root>/<folder_name>/<class>/<image>.

    The full folder is split 70/10/20 into train/val/test with split_seed.
    """
    big = cfg.datasets.big
    size = int(big.image_size)
    steps = [transforms.Resize((size, size))]
    if big.grayscale:
        steps.append(transforms.Grayscale(1))
    steps.append(transforms.ToTensor())
    folder = root / big.folder_name
    if not folder.exists():
        raise FileNotFoundError(
            f"Expected images in {folder}/<class_name>/*.jpg. "
            "Download the dataset and arrange it in this layout first."
        )
    full = datasets.ImageFolder(folder, transform=transforms.Compose(steps))
    n = len(full)
    n_test = int(0.2 * n)
    n_val = int(0.1 * n)
    gen = torch.Generator().manual_seed(cfg.project.split_seed)
    train, val, test = random_split(full, [n - n_val - n_test, n_val, n_test], generator=gen)
    channels = 1 if big.grayscale else 3
    return DatasetBundle(big.folder_name, train, val, test, list(full.classes), channels, size)


def load_dataset(which: str, cfg: Config) -> DatasetBundle:
    """which = 'small' or 'big' (as in the config), or a dataset name directly."""
    name = cfg.datasets[which]["name"] if which in ("small", "big") else which
    root = data_path(cfg)
    loaders = {"fashion_mnist": _fashion_mnist, "cifar10": _cifar10, "image_folder": _image_folder}
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(loaders)}")
    return loaders[name](root, cfg)


def make_loaders(bundle: DatasetBundle, cfg: Config) -> dict[str, DataLoader]:
    bs = cfg.datasets.batch_size
    nw = cfg.datasets.num_workers
    return {
        "train": DataLoader(bundle.train, batch_size=bs, shuffle=True, num_workers=nw),
        "val": DataLoader(bundle.val, batch_size=bs, shuffle=False, num_workers=nw),
        "test": DataLoader(bundle.test, batch_size=bs, shuffle=False, num_workers=nw),
    }


def to_numpy(ds: Dataset, limit: int | None = None, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with X flattened to (N, C*H*W) for scikit-learn models."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    xs, ys, seen = [], [], 0
    for x, y in loader:
        xs.append(x.reshape(x.shape[0], -1).numpy())
        ys.append(y.numpy())
        seen += len(y)
        if limit and seen >= limit:
            break
    X, y = np.concatenate(xs), np.concatenate(ys)
    return (X[:limit], y[:limit]) if limit else (X, y)
