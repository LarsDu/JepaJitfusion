"""Tests for data transforms and multi-crop augmentation."""

import torch
from PIL import Image

from jepajitfusion.data.transforms import forward_transform, reverse_transform
from jepajitfusion.encoder.multicrop import MultiCropAugmentation


def test_forward_transform_shape():
    transform = forward_transform(64)
    img = Image.new("RGB", (96, 96), color=(128, 128, 128))
    tensor = transform(img)
    assert tensor.shape == (3, 64, 64)
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_forward_transform_range():
    """Output should be in [-1, 1]."""
    transform = forward_transform(64)
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    t = transform(img)
    assert t.min() >= -1.0
    img_white = Image.new("RGB", (64, 64), color=(255, 255, 255))
    t_white = transform(img_white)
    assert t_white.max() <= 1.0


def test_reverse_transform():
    rev = reverse_transform()
    tensor = torch.zeros(3, 64, 64)  # value 0 → pixel 127.5
    img = rev(tensor)
    assert img.size == (64, 64)


def test_multicrop_augmentation_counts():
    mc = MultiCropAugmentation(
        n_global=2, n_local=4,
        global_size=64, local_size=32,
    )
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    crops = mc(img)
    assert len(crops) == 6  # 2 global + 4 local


def test_multicrop_augmentation_sizes():
    mc = MultiCropAugmentation(
        n_global=2, n_local=3,
        global_size=64, local_size=32,
    )
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    crops = mc(img)

    # Global crops should be 64x64
    for crop in crops[:2]:
        assert crop.shape == (3, 64, 64)

    # Local crops should be 32x32
    for crop in crops[2:]:
        assert crop.shape == (3, 32, 32)


def test_multicrop_normalize_range():
    mc = MultiCropAugmentation(n_global=1, n_local=0, global_size=64)
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    crops = mc(img)
    assert crops[0].min() >= -1.0
    assert crops[0].max() <= 1.0
