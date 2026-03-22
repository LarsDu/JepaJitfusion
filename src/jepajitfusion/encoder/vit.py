"""Vision Transformer encoder for LeJEPA.

Standard ViT with pre-norm, learned positional embeddings, and CLS token.
Supports variable input sizes via positional embedding interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Embed image patches using a Conv2d projection."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)  # (B, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual, LN → MLP → residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT encoder with CLS token and learned positional embeddings.

    Returns the CLS token embedding by default. Supports multi-scale input
    via positional embedding interpolation (for multi-crop SSL).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _get_pos_embed(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings, interpolating if input size differs."""
        if seq_len == self.pos_embed.shape[1]:
            return self.pos_embed

        cls_embed = self.pos_embed[:, :1, :]
        patch_embed = self.pos_embed[:, 1:, :]

        N_orig = patch_embed.shape[1]
        h_orig = w_orig = int(N_orig**0.5)

        N_new = seq_len - 1
        h_new = w_new = int(N_new**0.5)

        patch_embed = patch_embed.reshape(1, h_orig, w_orig, -1).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(
            patch_embed.float(),
            size=(h_new, w_new),
            mode="bicubic",
            align_corners=False,
        )
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
        return torch.cat([cls_embed, patch_embed], dim=1)

    def forward(
        self, x: torch.Tensor, return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images.
            return_all_tokens: if True, return (B, N+1, D) instead of (B, D).

        Returns:
            CLS embedding (B, D) or all tokens (B, N+1, D).
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = x + self._get_pos_embed(N + 1).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        if return_all_tokens:
            return x
        return x[:, 0]
