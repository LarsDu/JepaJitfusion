"""Tests for data transforms and multi-crop augmentation."""

import torch
from PIL import Image

from jepajitfusion.data.downloader import pokedex_id_from_filename
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


def test_pokedex_id_from_filename():
    # Plain Pokedex entries key on the leading digits.
    assert pokedex_id_from_filename("100_bw_m.png") == "100"
    assert pokedex_id_from_filename("1_rs_s.png") == "1"
    # Forme variants share the same Pokedex entry as the base number.
    assert pokedex_id_from_filename("351-rainy_bw_m_s.png") == "351"
    assert pokedex_id_from_filename("351-sunny_rs_s.png") == "351"
    assert pokedex_id_from_filename("386-attack_pt_m_f2.png") == "386"


def test_pokemon_split_has_no_pokedex_leakage():
    """No Pokedex entry may appear in both the train and test splits."""
    from pathlib import Path

    base = Path("downloads/pokemon_11k")
    train_dir = base / "train" / "class_0"
    test_dir = base / "test" / "class_0"
    if not (train_dir.exists() and test_dir.exists()):
        import pytest

        pytest.skip("pokemon_11k dataset not downloaded")

    train_ids = {pokedex_id_from_filename(p.name) for p in train_dir.glob("*.png")}
    test_ids = {pokedex_id_from_filename(p.name) for p in test_dir.glob("*.png")}
    assert train_ids and test_ids
    assert train_ids.isdisjoint(test_ids), sorted(train_ids & test_ids)
