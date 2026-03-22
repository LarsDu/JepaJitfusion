"""Tests for SIGReg loss."""

import torch

from jepajitfusion.encoder.sigreg import SIGReg, SlicingUnivariateTest, UnivariateGaussianityTest


def test_univariate_test_normal_input():
    """SIGReg loss should be near zero for standard normal input."""
    test = UnivariateGaussianityTest(t_max=3.0, n_quad=17)
    # Large batch of standard normal data
    z = torch.randn(10000, 8)  # 8 slices
    loss = test(z)
    assert loss.item() < 0.05, f"Loss for normal data should be small, got {loss.item()}"


def test_univariate_test_nonnormal_input():
    """SIGReg loss should be positive for non-normal input."""
    test = UnivariateGaussianityTest(t_max=3.0, n_quad=17)
    # Uniform distribution (not normal)
    z = torch.rand(10000, 8) * 4 - 2
    loss_uniform = test(z)

    # Normal distribution for comparison
    z_normal = torch.randn(10000, 8)
    loss_normal = test(z_normal)

    assert loss_uniform.item() > loss_normal.item(), (
        f"Uniform loss {loss_uniform.item()} should exceed normal loss {loss_normal.item()}"
    )


def test_slicing_test_shape():
    sut = SlicingUnivariateTest(embed_dim=256, n_slices=32)
    z = torch.randn(64, 256)
    loss = sut(z)
    assert loss.shape == ()  # scalar


def test_sigreg_forward():
    sigreg = SIGReg(embed_dim=256, n_slices=32)
    z1 = torch.randn(64, 256)
    z2 = torch.randn(64, 256)
    total_loss, metrics = sigreg(z1, z2)
    assert total_loss.shape == ()
    assert "invariance_loss" in metrics
    assert "regularization_loss" in metrics


def test_sigreg_identical_views_low_invariance():
    """Invariance loss should be near zero for identical embeddings."""
    sigreg = SIGReg(embed_dim=256, n_slices=32, invariance_weight=1.0, regularization_weight=0.0)
    z = torch.randn(64, 256)
    loss, metrics = sigreg(z, z)  # identical views
    assert metrics["invariance_loss"] < 1e-6


def test_sigreg_gradient_flow():
    sigreg = SIGReg(embed_dim=64, n_slices=16)
    z1 = torch.randn(32, 64, requires_grad=True)
    z2 = torch.randn(32, 64, requires_grad=True)
    loss, _ = sigreg(z1, z2)
    loss.backward()
    assert z1.grad is not None
    assert z2.grad is not None
