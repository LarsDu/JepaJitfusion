"""Tests for the ViT encoder."""

import torch

from jepajitfusion.encoder.vit import (
    Attention,
    MLP,
    PatchEmbed,
    TransformerBlock,
    VisionTransformer,
)


def test_patch_embed_shape():
    pe = PatchEmbed(img_size=64, patch_size=8, in_channels=3, embed_dim=256)
    x = torch.randn(2, 3, 64, 64)
    out = pe(x)
    assert out.shape == (2, 64, 256)  # 64 patches = (64/8)^2


def test_attention_shape():
    attn = Attention(dim=256, num_heads=4)
    x = torch.randn(2, 65, 256)  # 64 patches + 1 CLS
    out = attn(x)
    assert out.shape == (2, 65, 256)


def test_mlp_shape():
    mlp = MLP(dim=256, mlp_ratio=4.0)
    x = torch.randn(2, 65, 256)
    out = mlp(x)
    assert out.shape == (2, 65, 256)


def test_transformer_block_shape():
    block = TransformerBlock(dim=256, num_heads=4, mlp_ratio=4.0)
    x = torch.randn(2, 65, 256)
    out = block(x)
    assert out.shape == (2, 65, 256)


def test_vision_transformer_cls_output():
    vit = VisionTransformer(
        img_size=64, patch_size=8, in_channels=3,
        embed_dim=256, depth=2, num_heads=4,
    )
    x = torch.randn(2, 3, 64, 64)
    cls = vit(x)
    assert cls.shape == (2, 256)


def test_vision_transformer_all_tokens():
    vit = VisionTransformer(
        img_size=64, patch_size=8, in_channels=3,
        embed_dim=256, depth=2, num_heads=4,
    )
    x = torch.randn(2, 3, 64, 64)
    all_tokens = vit(x, return_all_tokens=True)
    assert all_tokens.shape == (2, 65, 256)  # 64 patches + 1 CLS


def test_vision_transformer_multiscale():
    """Test that the encoder handles different input sizes via pos_embed interpolation."""
    vit = VisionTransformer(
        img_size=64, patch_size=8, in_channels=3,
        embed_dim=256, depth=2, num_heads=4,
    )
    # 32x32 input (local crop size)
    x_small = torch.randn(2, 3, 32, 32)
    cls_small = vit(x_small)
    assert cls_small.shape == (2, 256)

    # 64x64 input (global crop size)
    x_large = torch.randn(2, 3, 64, 64)
    cls_large = vit(x_large)
    assert cls_large.shape == (2, 256)


def test_vision_transformer_gradient_flow():
    vit = VisionTransformer(
        img_size=64, patch_size=8, in_channels=3,
        embed_dim=256, depth=2, num_heads=4,
    )
    x = torch.randn(2, 3, 64, 64)
    out = vit(x)
    loss = out.sum()
    loss.backward()
    # Check that gradients flow through the transformer
    # Test on qkv weights which are directly in the attention path
    assert vit.blocks[0].attn.qkv.weight.grad is not None
    assert vit.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
