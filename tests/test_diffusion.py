"""Tests for diffusion utilities."""

import torch

from jepajitfusion.decoder.diffusion import (
    compute_v_loss,
    compute_z_t,
    sample_logit_normal_time,
)


def test_logit_normal_time_range():
    """Logit-normal samples should be in (0, 1)."""
    t = sample_logit_normal_time(1000, P_mean=-0.8, P_std=0.8)
    assert t.min() > 0
    assert t.max() < 1
    assert t.shape == (1000,)


def test_logit_normal_time_distribution():
    """Mean should be approximately sigmoid(P_mean)."""
    t = sample_logit_normal_time(100000, P_mean=-0.8, P_std=0.8)
    expected_mean = torch.sigmoid(torch.tensor(-0.8)).item()
    actual_mean = t.mean().item()
    assert abs(actual_mean - expected_mean) < 0.05


def test_z_t_at_boundaries():
    """z_t at t=1 should be x, at t=0 should be noise."""
    x = torch.randn(4, 3, 64, 64)
    noise = torch.randn(4, 3, 64, 64)

    # t=1 → z_t = x
    t1 = torch.ones(4)
    z1 = compute_z_t(x, noise, t1)
    assert torch.allclose(z1, x, atol=1e-6)

    # t=0 → z_t = noise
    t0 = torch.zeros(4)
    z0 = compute_z_t(x, noise, t0)
    assert torch.allclose(z0, noise, atol=1e-6)


def test_z_t_interpolation():
    """z_t at t=0.5 should be midpoint."""
    x = torch.ones(4, 3, 8, 8)
    noise = torch.zeros(4, 3, 8, 8)
    t = torch.full((4,), 0.5)
    z = compute_z_t(x, noise, t)
    assert torch.allclose(z, torch.full_like(z, 0.5), atol=1e-6)


def test_v_loss_runs():
    """v-loss computation should run without error and return scalar."""
    from jepajitfusion.decoder.jit_model import JiTModel

    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=64, depth=1, num_heads=2,
        bottleneck_dim=16, conditioning_mode="none",
    )
    x = torch.randn(2, 3, 64, 64)
    t = sample_logit_normal_time(2)
    noise = torch.randn_like(x)
    loss = compute_v_loss(model, x, t, noise, noise_scale=0.25)
    assert loss.shape == ()
    assert loss.item() > 0
