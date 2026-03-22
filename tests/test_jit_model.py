"""Tests for the JiT decoder model."""

import torch

from jepajitfusion.decoder.jit_model import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTBlock,
    JiTModel,
    RMSNorm,
    SwiGLU,
    VisionRoPE,
)


def test_rmsnorm():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_swiglu():
    ffn = SwiGLU(dim=384, mlp_ratio=4.0)
    x = torch.randn(2, 256, 384)
    out = ffn(x)
    assert out.shape == (2, 256, 384)


def test_vision_rope():
    rope = VisionRoPE(head_dim=64, num_patches_h=16, num_patches_w=16)
    q = torch.randn(2, 6, 256, 64)  # (B, heads, N, head_dim)
    k = torch.randn(2, 6, 256, 64)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_bottleneck_patch_embed():
    pe = BottleneckPatchEmbed(
        img_size=64, patch_size=4, in_channels=3, embed_dim=384, bottleneck_dim=64
    )
    x = torch.randn(2, 3, 64, 64)
    out = pe(x)
    assert out.shape == (2, 256, 384)  # 256 patches = (64/4)^2


def test_jit_block():
    block = JiTBlock(dim=384, num_heads=6, mlp_ratio=4.0)
    rope = VisionRoPE(head_dim=64, num_patches_h=16, num_patches_w=16)
    x = torch.randn(2, 256, 384)
    c = torch.randn(2, 384)
    out = block(x, c, rope)
    assert out.shape == (2, 256, 384)


def test_jit_block_identity_init():
    """Gates should be zero-initialized, so block output ≈ input initially."""
    block = JiTBlock(dim=384, num_heads=6)
    rope = VisionRoPE(head_dim=64, num_patches_h=16, num_patches_w=16)
    x = torch.randn(2, 256, 384)
    c = torch.zeros(2, 384)  # zero conditioning
    out = block(x, c, rope)
    # Output should be close to input (identity function with zero gates)
    diff = (out - x).abs().max().item()
    assert diff < 1e-4, f"Block should be near-identity at init, max diff={diff}"


def test_final_layer():
    fl = FinalLayer(dim=384, patch_size=4, in_channels=3)
    x = torch.randn(2, 256, 384)
    c = torch.randn(2, 384)
    out = fl(x, c)
    assert out.shape == (2, 256, 48)  # 4*4*3 = 48


def test_jit_model_unconditional():
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=384, depth=2, num_heads=6,
        bottleneck_dim=64, conditioning_mode="none",
    )
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == (2, 3, 64, 64)


def test_jit_model_label_conditioned():
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=384, depth=2, num_heads=6,
        bottleneck_dim=64, num_classes=10, conditioning_mode="label",
    )
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    labels = torch.randint(0, 10, (2,))
    out = model(x, t, labels)
    assert out.shape == (2, 3, 64, 64)


def test_jit_model_jepa_conditioned():
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=384, depth=2, num_heads=6,
        bottleneck_dim=64, conditioning_mode="jepa", jepa_dim=256,
    )
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    jepa_emb = torch.randn(2, 256)
    out = model(x, t, jepa_emb)
    assert out.shape == (2, 3, 64, 64)


def test_jit_model_gradient_flow():
    model = JiTModel(
        img_size=64, patch_size=4, in_channels=3,
        dim=384, depth=2, num_heads=6,
        bottleneck_dim=64, conditioning_mode="none",
    )
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    out = model(x, t)
    loss = out.sum()
    loss.backward()
    assert model.patch_embed.proj[0].weight.grad is not None
