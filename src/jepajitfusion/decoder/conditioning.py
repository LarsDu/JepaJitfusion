"""Conditioning modules for the JiT decoder.

TimestepEmbedder: sinusoidal → MLP for diffusion timesteps
LabelEmbedder: class label embedding with CFG dropout
JepaConditioner: projects LeJEPA CLS embeddings into JiT's adaLN space
"""

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps via sinusoidal encoding -> MLP."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal positional embedding from scalar timesteps.

        Args:
            t: (B,) timestep values.
            dim: Embedding dimension.

        Returns:
            (B, dim) sinusoidal embedding.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep values in [0, 1].

        Returns:
            (B, dim) conditioning vector.
        """
        emb = self.sinusoidal_embedding(t, self.freq_dim)
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    """Class label embedding with dropout for classifier-free guidance.

    During training, labels are randomly replaced with a null token
    (index=num_classes) with probability dropout_prob.
    """

    def __init__(self, num_classes: int, dim: int, dropout_prob: float = 0.1):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding = nn.Embedding(num_classes + int(use_cfg), dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: (B,) integer class labels.

        Returns:
            (B, dim) label embeddings.
        """
        if self.training and self.dropout_prob > 0:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_mask, self.num_classes, labels)
        return self.embedding(labels)


class JepaConditioner(nn.Module):
    """Projects LeJEPA CLS embeddings into JiT's adaLN conditioning space.

    Replaces the LabelEmbedder in fusion mode.
    """

    def __init__(self, jepa_dim: int, jit_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(jepa_dim, jit_dim),
            nn.SiLU(),
            nn.Linear(jit_dim, jit_dim),
        )

    def forward(self, jepa_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            jepa_embedding: (B, jepa_dim) CLS token from LeJEPA encoder.

        Returns:
            (B, jit_dim) conditioning vector.
        """
        return self.proj(jepa_embedding)
