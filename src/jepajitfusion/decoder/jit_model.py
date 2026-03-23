"""JiT (Just Image Transformers) decoder model.

A plain ViT for diffusion with:
- Bottleneck patch embedding
- RMSNorm instead of LayerNorm
- SwiGLU FFN
- 2D Rotary Position Embeddings (RoPE)
- adaLN-Zero conditioning (shift, scale, gate per block)
- x-prediction (predicts clean image directly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepajitfusion.decoder.conditioning import (
    JepaConditioner,
    LabelEmbedder,
    TimestepEmbedder,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: SiLU(xW1) * xW2 → W3."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        # Round to nearest multiple of 8 for efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class VisionRoPE(nn.Module):
    """2D Rotary Position Embeddings for vision transformers.

    Splits head_dim into two halves:
    - First half encodes row (height) position
    - Second half encodes column (width) position
    """

    def __init__(
        self,
        head_dim: int,
        num_patches_h: int,
        num_patches_w: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

        quarter_dim = head_dim // 4
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, quarter_dim, dtype=torch.float32) / quarter_dim
            )
        )

        h_pos = torch.arange(num_patches_h, dtype=torch.float32)
        w_pos = torch.arange(num_patches_w, dtype=torch.float32)

        # Outer products: position × frequency
        freqs_h = torch.outer(h_pos, freqs)  # (H, quarter_dim)
        freqs_w = torch.outer(w_pos, freqs)  # (W, quarter_dim)

        # Expand to full 2D grid
        freqs_h = freqs_h[:, None, :].expand(-1, num_patches_w, -1)  # (H, W, qd)
        freqs_w = freqs_w[None, :, :].expand(num_patches_h, -1, -1)  # (H, W, qd)

        # Concatenate h and w frequencies, flatten spatial
        freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1)  # (H, W, half_dim)
        freqs_2d = freqs_2d.reshape(-1, head_dim // 2)  # (N, half_dim)

        self.register_buffer("cos_cached", freqs_2d.cos())  # (N, half_dim)
        self.register_buffer("sin_cached", freqs_2d.sin())  # (N, half_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE to queries and keys.

        Args:
            q, k: (B, heads, N, head_dim)

        Returns:
            Rotated (q, k) with same shape.
        """
        return self._rotate(q), self._rotate(k)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        cos = self.cos_cached[None, None, :, :]  # (1, 1, N, half_dim)
        sin = self.sin_cached[None, None, :, :]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class BottleneckPatchEmbed(nn.Module):
    """Patch embedding with bottleneck: patches → small dim → model dim.

    Reduces the initial dimensionality of raw patches before projecting
    up to the model dimension.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 384,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Sequential(
            nn.Linear(patch_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, N, embed_dim) where N = num_patches
        """
        B, C, H, W = x.shape
        pH = self.num_patches_h
        pW = self.num_patches_w
        ps = self.patch_size

        # Reshape into patches
        x = x.reshape(B, C, pH, ps, pW, ps)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, pH, pW, C, ps, ps)
        x = x.reshape(B, pH * pW, -1)  # (B, N, patch_dim)
        return self.proj(x)


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, rope: VisionRoPE | None = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)

        if rope is not None:
            q, k = rope(q, k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class JiTBlock(nn.Module):
    """JiT transformer block with adaLN-Zero conditioning.

    Per block, the conditioning vector c produces 6 modulation parameters
    (shift, scale, gate for attention and FFN). Gates are initialized to
    zero so each block starts as an identity function.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, mlp_ratio=mlp_ratio)

        # adaLN modulation: c → 6 * dim params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init gates for identity initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, rope: VisionRoPE | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim) patch tokens.
            c: (B, dim) conditioning vector.
            rope: Optional VisionRoPE for positional encoding.

        Returns:
            (B, N, dim) output tokens.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Attention branch
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, rope)
        x = x + gate_msa.unsqueeze(1) * h

        # FFN branch
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.ffn(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear projection back to patch pixel space."""

    def __init__(self, dim: int, patch_size: int, in_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self.linear = nn.Linear(dim, patch_size * patch_size * in_channels)

        # Zero initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class JiTModel(nn.Module):
    """JiT (Just Image Transformers) diffusion model.

    A plain ViT that directly predicts clean images (x-prediction) from
    noisy inputs, using flow-matching with adaLN-Zero conditioning.

    Supports three conditioning modes:
    - "none": unconditional generation
    - "label": class-conditional with CFG dropout
    - "jepa": conditioned on LeJEPA encoder embeddings
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        bottleneck_dim: int = 64,
        num_classes: int = 0,
        conditioning_mode: str = "none",
        jepa_dim: int = 256,
    ):
        super().__init__()
        self.conditioning_mode = conditioning_mode
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size

        # Patch embedding
        self.patch_embed = BottleneckPatchEmbed(
            img_size, patch_size, in_channels, dim, bottleneck_dim
        )

        # Positional encoding via RoPE
        head_dim = dim // num_heads
        self.rope = VisionRoPE(
            head_dim, self.num_patches_h, self.num_patches_w
        )

        # Conditioning
        self.time_embed = TimestepEmbedder(dim)
        if conditioning_mode == "label" and num_classes > 0:
            self.label_embed = LabelEmbedder(num_classes, dim)
        elif conditioning_mode == "jepa":
            self.jepa_cond = JepaConditioner(jepa_dim, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [JiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # Final projection
        self.final_layer = FinalLayer(dim, patch_size, in_channels)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"JiT model: {n_params:.1f}M parameters, mode={conditioning_mode}")

    def _init_weights(self):
        # Initialize patch embed and transformer weights
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.patch_embed.apply(_init)
        self.blocks.apply(_init)
        # TimestepEmbedder, LabelEmbedder, JepaConditioner use default init

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch predictions back to image tensor.

        Args:
            x: (B, N, patch_size^2 * C)

        Returns:
            (B, C, H, W)
        """
        B = x.shape[0]
        pH, pW = self.num_patches_h, self.num_patches_w
        ps = self.patch_size
        C = self.in_channels

        x = x.reshape(B, pH, pW, C, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, pH, ps, pW, ps)
        return x.reshape(B, C, pH * ps, pW * ps)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image z_t.
            t: (B,) timestep values.
            y: Optional conditioning — class labels (LongTensor) or
               JEPA embeddings (FloatTensor), depending on mode.

        Returns:
            (B, C, H, W) predicted clean image x_pred.
        """
        # Patch embed
        x = self.patch_embed(x)  # (B, N, dim)

        # Build conditioning vector
        c = self.time_embed(t)  # (B, dim)
        if self.conditioning_mode == "label" and y is not None:
            c = c + self.label_embed(y)
        elif self.conditioning_mode == "jepa" and y is not None:
            c = c + self.jepa_cond(y)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, self.rope)

        # Final layer + unpatchify
        x = self.final_layer(x, c)
        return self.unpatchify(x)
