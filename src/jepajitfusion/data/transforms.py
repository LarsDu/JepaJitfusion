"""Image transforms for training and visualization."""

from typing import Callable, Sequence

import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)


def _normalize_to_neg1_1(t: torch.Tensor) -> torch.Tensor:
    return (t * 2) - 1


def _denormalize_to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t + 1) / 2).clamp(0, 1) * 255.0


def _to_hwc_numpy(t: torch.Tensor):
    return t.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)


def forward_transform(img_size: int | Sequence[int] = 64) -> Callable:
    """Standard training transform: resize, crop, normalize to [-1, 1]."""
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    return Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            _normalize_to_neg1_1,
            RandomHorizontalFlip(),
        ]
    )


def reverse_transform() -> Callable:
    """Convert an unbatched [-1,1] tensor back to a PIL image."""
    return Compose(
        [
            _denormalize_to_uint8,
            _to_hwc_numpy,
            ToPILImage(),
        ]
    )
