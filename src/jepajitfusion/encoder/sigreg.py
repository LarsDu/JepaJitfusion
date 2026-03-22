"""SIGReg loss: Epps-Pulley characteristic function test for SSL regularization.

The SIGReg loss has two components:
  1. Invariance: MSE between L2-normalized embeddings of different views
  2. Regularization: SlicingUnivariateTest ensures the embedding distribution
     matches a standard normal via the Epps-Pulley characteristic function test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnivariateGaussianityTest(nn.Module):
    """Tests if 1D data follows N(0,1) using the Epps-Pulley characteristic function test.

    Computes: sum_k w_k * (phi_emp(t_k) - phi_N(t_k))^2
    where phi_emp is the empirical characteristic function and
    phi_N is the standard normal CF.
    """

    def __init__(self, t_max: float = 3.0, n_quad: int = 17):
        super().__init__()
        t_points = torch.linspace(0, t_max, n_quad)
        self.register_buffer("t_points", t_points)
        # Gaussian-windowed weights
        self.register_buffer("weights", torch.exp(-t_points**2 / 2))
        # Standard normal CF: phi_N(t) = exp(-t^2/2)
        self.register_buffer("phi_normal", torch.exp(-t_points**2 / 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, S) — B samples, S slices. Each column is a 1D sample.

        Returns:
            Scalar loss averaged over slices.
        """
        # z: (B, S), t_points: (Q,)
        # Compute cos(t_k * z_j) for all samples, slices, and quadrature points
        # z.unsqueeze(-1): (B, S, 1), t_points: (1, 1, Q)
        tz = z.unsqueeze(-1) * self.t_points.unsqueeze(0).unsqueeze(0)  # (B, S, Q)
        # Empirical CF (real part, by symmetry): mean over batch
        phi_emp = torch.cos(tz).mean(dim=0)  # (S, Q)

        # Squared difference weighted by Gaussian window
        diff = phi_emp - self.phi_normal.unsqueeze(0)  # (S, Q)
        loss = (self.weights.unsqueeze(0) * diff**2).sum(dim=-1).mean()
        return loss


class SlicingUnivariateTest(nn.Module):
    """Projects D-dim embeddings to random 1D slices and tests each for Gaussianity.

    If Z ~ N(0, I), then for any unit vector u, u^T Z ~ N(0, 1).
    We test this property via random projections.
    """

    def __init__(
        self,
        embed_dim: int,
        n_slices: int = 64,
        t_max: float = 3.0,
        n_quad: int = 17,
    ):
        super().__init__()
        self.test = UnivariateGaussianityTest(t_max, n_quad)
        # Fixed random unit vectors for projection
        directions = torch.randn(embed_dim, n_slices)
        directions = F.normalize(directions, dim=0)
        self.register_buffer("directions", directions)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) — batch of embeddings.

        Returns:
            Scalar loss.
        """
        # Center and standardize across batch (per-dimension)
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, keepdim=True) + 1e-6)

        # Project to random 1D slices
        projections = z @ self.directions  # (B, n_slices)

        return self.test(projections)


class SIGReg(nn.Module):
    """SIGReg loss combining invariance and Gaussianity regularization.

    Args:
        embed_dim: Dimension of the embeddings.
        n_slices: Number of random projection directions.
        t_max: Max quadrature point for CF test.
        n_quad: Number of quadrature points.
        invariance_weight: Weight for the invariance (MSE) term.
        regularization_weight: Weight for the regularization (Gaussianity) term.
    """

    def __init__(
        self,
        embed_dim: int,
        n_slices: int = 64,
        t_max: float = 3.0,
        n_quad: int = 17,
        invariance_weight: float = 25.0,
        regularization_weight: float = 1.0,
    ):
        super().__init__()
        self.invariance_weight = invariance_weight
        self.regularization_weight = regularization_weight
        self.slicing_test = SlicingUnivariateTest(embed_dim, n_slices, t_max, n_quad)

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            z1, z2: (B, D) — embeddings from two views of the same images.

        Returns:
            (total_loss, metrics_dict)
        """
        # Invariance: MSE between L2-normalized embeddings
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        inv_loss = F.mse_loss(z1_norm, z2_norm)

        # Regularization: Gaussianity test on each view
        reg_loss = 0.5 * (self.slicing_test(z1) + self.slicing_test(z2))

        total = self.invariance_weight * inv_loss + self.regularization_weight * reg_loss

        metrics = {
            "invariance_loss": inv_loss.item(),
            "regularization_loss": reg_loss.item(),
            "total_loss": total.item(),
        }
        return total, metrics
