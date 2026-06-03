"""MLP projection head for SSL."""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Three-layer MLP projection head used in leJEPA SSL training.

    Projects the encoder output to the lower-dimensional space where the SIGReg
    loss is computed. Follows the reference leJEPA projector: each hidden layer is
    Linear -> BatchNorm1d -> GELU, and the final layer is a plain Linear (no norm,
    no activation) so SIGReg sees an unconstrained representation in R^out_dim.
    BatchNorm stabilizes the projected statistics across the batch.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
