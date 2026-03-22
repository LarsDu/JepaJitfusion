"""Tests for conditioning modules."""

import torch

from jepajitfusion.decoder.conditioning import (
    JepaConditioner,
    LabelEmbedder,
    TimestepEmbedder,
)


def test_timestep_embedder_shape():
    te = TimestepEmbedder(dim=384)
    t = torch.rand(4)
    out = te(t)
    assert out.shape == (4, 384)


def test_timestep_embedder_different_timesteps():
    """Different timesteps should produce different embeddings."""
    te = TimestepEmbedder(dim=384)
    t1 = torch.tensor([0.1])
    t2 = torch.tensor([0.9])
    e1 = te(t1)
    e2 = te(t2)
    assert not torch.allclose(e1, e2)


def test_label_embedder_shape():
    le = LabelEmbedder(num_classes=10, dim=384)
    labels = torch.randint(0, 10, (4,))
    out = le(labels)
    assert out.shape == (4, 384)


def test_label_embedder_cfg_dropout():
    """In training mode with high dropout, some labels should be replaced."""
    le = LabelEmbedder(num_classes=10, dim=384, dropout_prob=0.5)
    le.train()
    labels = torch.zeros(100, dtype=torch.long)
    out = le(labels)
    assert out.shape == (100, 384)


def test_jepa_conditioner_shape():
    jc = JepaConditioner(jepa_dim=256, jit_dim=384)
    emb = torch.randn(4, 256)
    out = jc(emb)
    assert out.shape == (4, 384)


def test_jepa_conditioner_gradient_flow():
    jc = JepaConditioner(jepa_dim=256, jit_dim=384)
    emb = torch.randn(4, 256, requires_grad=True)
    out = jc(emb)
    out.sum().backward()
    assert emb.grad is not None
