"""Tests for color augmentations used in the morphology experiments."""

import pickle

import torch

from jepajitfusion.encoder.color_transforms import (
    LabColorTransform,
    build_color_augmentation,
    lab_to_rgb,
    rgb_to_lab,
)
from jepajitfusion.encoder.multicrop import MultiCropAugmentation


def _random_img(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, 32, 32, generator=g)


def test_lab_roundtrip_is_identity():
    """rgb -> lab -> rgb recovers the original image within tolerance."""
    img = _random_img()
    out = lab_to_rgb(rgb_to_lab(img))
    assert torch.allclose(img, out, atol=1e-4), (img - out).abs().max().item()


def test_hue_rotation_preserves_lab_distances():
    """A pure chroma-plane rotation is an isometry in Lab: pairwise color distances
    between pixels are preserved (this is the relational-preservation property)."""
    img = _random_img(1)
    lab = rgb_to_lab(img)
    # Rotate (a, b) by 73 degrees; leave L untouched. No chroma scaling.
    theta = torch.tensor(73.0) * torch.pi / 180.0
    c, s = torch.cos(theta), torch.sin(theta)
    L, a, b = lab[0], lab[1], lab[2]
    lab_rot = torch.stack([L, a * c - b * s, a * s + b * c], dim=0)

    px = lab.reshape(3, -1).T[:200].double()  # (N, 3)
    px_rot = lab_rot.reshape(3, -1).T[:200].double()
    # Direct pairwise distances in float64 (cdist's expanded formula is unstable).
    d0 = (px[:, None, :] - px[None, :, :]).norm(dim=-1)
    d1 = (px_rot[:, None, :] - px_rot[None, :, :]).norm(dim=-1)
    assert torch.allclose(d0, d1, atol=1e-4), (d0 - d1).abs().max().item()


def test_lab_transform_output_range_and_shape():
    img = _random_img(2)
    t = LabColorTransform(hue_deg=45.0, lightness=0.1, chroma=0.3)
    out = t(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_lab_transform_zero_params_is_near_identity():
    """With no hue/lightness/chroma jitter the transform is ~identity (roundtrip)."""
    img = _random_img(3)
    t = LabColorTransform(hue_deg=0.0, lightness=0.0, chroma=0.0)
    out = t(img)
    assert torch.allclose(img, out, atol=1e-4)


def test_build_color_augmentation_modes():
    pil_d, ten_d = build_color_augmentation("default")
    assert len(pil_d) == 1 and ten_d == []
    pil_cj, ten_cj = build_color_augmentation("colorjitter", hue=0.2)
    assert len(pil_cj) == 1 and ten_cj == []
    pil_lab, ten_lab = build_color_augmentation("lab")
    assert pil_lab == [] and len(ten_lab) == 1
    try:
        build_color_augmentation("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_multicrop_lab_picklable_and_runs():
    """Lab multicrop must be picklable (DataLoader workers) and produce crops."""
    from PIL import Image

    mc = MultiCropAugmentation(
        n_global=2, n_local=2, global_size=32, local_size=16, color_aug="lab"
    )
    mc2 = pickle.loads(pickle.dumps(mc))
    img = Image.fromarray((torch.rand(48, 48, 3) * 255).byte().numpy())
    crops = mc2(img)
    assert len(crops) == 4
    assert crops[0].shape == (3, 32, 32)
    assert crops[2].shape == (3, 16, 16)
    # Normalized to [-1, 1].
    assert crops[0].min() >= -1.001 and crops[0].max() <= 1.001
