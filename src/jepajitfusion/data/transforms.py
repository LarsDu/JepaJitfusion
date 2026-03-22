"""Image transforms for training and visualization."""

from typing import Callable, Sequence

import numpy as np
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomHorizontalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)


def forward_transform(img_size: int | Sequence[int] = 64) -> Callable:
    """Standard training transform: resize, crop, normalize to [-1, 1]."""
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    return Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),
            RandomHorizontalFlip(),
        ]
    )


def reverse_transform() -> Callable:
    """Convert an unbatched [-1,1] tensor back to a PIL image."""
    return Compose(
        [
            Lambda(lambda t: ((t + 1) / 2).clamp(0, 1) * 255.0),
            Lambda(
                lambda t: t.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
            ),
            ToPILImage(),
        ]
    )
