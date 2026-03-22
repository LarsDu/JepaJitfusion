"""Tests for the Heun ODE sampler."""

import torch

from jepajitfusion.decoder.jit_model import JiTModel
from jepajitfusion.decoder.sampler import HeunSampler


def test_heun_sampler_shape():
    """Sampler should produce output of correct shape."""
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=64, depth=1, num_heads=2,
        bottleneck_dim=16, conditioning_mode="none",
    )
    model.eval()

    sampler = HeunSampler(num_steps=5, cfg_scale=1.0, noise_scale=0.25)
    shape = (2, 3, 64, 64)
    device = torch.device("cpu")

    samples = sampler.sample(model, shape, device)
    assert samples.shape == shape


def test_heun_sampler_deterministic():
    """Same seed should produce same samples."""
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=64, depth=1, num_heads=2,
        bottleneck_dim=16, conditioning_mode="none",
    )
    model.eval()

    sampler = HeunSampler(num_steps=3, cfg_scale=1.0, noise_scale=0.25)
    shape = (2, 3, 64, 64)
    device = torch.device("cpu")

    torch.manual_seed(42)
    s1 = sampler.sample(model, shape, device)

    torch.manual_seed(42)
    s2 = sampler.sample(model, shape, device)

    assert torch.allclose(s1, s2)


def test_heun_sampler_with_label_conditioning():
    """Sampler should work with label conditioning."""
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=64, depth=1, num_heads=2,
        bottleneck_dim=16, num_classes=10, conditioning_mode="label",
    )
    model.eval()

    sampler = HeunSampler(num_steps=3, cfg_scale=1.5, noise_scale=0.25)
    shape = (2, 3, 64, 64)
    device = torch.device("cpu")

    labels = torch.tensor([0, 5])
    uncond_labels = torch.full((2,), 10, dtype=torch.long)  # null class

    samples = sampler.sample(
        model, shape, device,
        conditioning=labels,
        uncond_conditioning=uncond_labels,
    )
    assert samples.shape == shape
