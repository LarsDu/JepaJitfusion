"""Dataset registry and DataLoader factory."""

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from jepajitfusion.data.downloader import (
    download_imagenette,
    download_pokemon_64,
    download_tiny_imagenet,
)

DATASET_REGISTRY = {
    "pokemon_64": download_pokemon_64,
    "imagenet_tiny": download_tiny_imagenet,
    "imagenette": download_imagenette,
}


def get_dataset(
    name: str,
    transform: Callable | None = None,
    data_dir: str = "downloads",
    test_size: float = 0.15,
) -> tuple[ImageFolder, ImageFolder]:
    """Get train/test datasets by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}")
    download_fn = DATASET_REGISTRY[name]
    return download_fn(transform=transform, data_dir=data_dir, test_size=test_size)


class MultiCropDataset(Dataset):
    """Wraps a dataset to return multiple augmented crops per image.

    The base dataset should return (PIL_Image, label) — pass transform=None
    to the underlying ImageFolder.
    """

    def __init__(self, base_dataset: Dataset, multicrop_transform: Callable):
        self.base_dataset = base_dataset
        self.multicrop = multicrop_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        crops = self.multicrop(img)  # list of tensors
        return crops, label


def multicrop_collate(batch):
    """Collate function for MultiCropDataset.

    Returns: (list[Tensor], Tensor) where each Tensor in the list is (B, C, H, W)
    for one crop position, and the second Tensor is labels (B,).
    """
    crops_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    n_crops = len(crops_list[0])
    batched_crops = [
        torch.stack([crops[i] for crops in crops_list]) for i in range(n_crops)
    ]
    return batched_crops, torch.tensor(labels, dtype=torch.long)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_fn,
    )
