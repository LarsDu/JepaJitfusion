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
        assert n_quad % 2 == 1, "n_quad must be odd"
        t_points = torch.linspace(0, t_max, n_quad)
        self.register_buffer("t_points", t_points)
        # Standard normal CF (real): phi_N(t) = exp(-t^2/2). Also the integration
        # window w(t) of the Epps-Pulley statistic.
        window = torch.exp(-t_points**2 / 2)
        self.register_buffer("phi_normal", window)
        # Trapezoid quadrature weights over [0, t_max]. Interior nodes get 2*dt
        # (the factor 2 accounts for the symmetric negative half [-t_max, 0] that we
        # do not evaluate explicitly); the two endpoints get dt. The window is folded
        # into the weights so the forward pass is a single weighted sum.
        dt = t_max / (n_quad - 1)
        quad = torch.full((n_quad,), 2 * dt)
        quad[0] = dt
        quad[-1] = dt
        self.register_buffer("weights", quad * window)

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
        # Empirical characteristic function: real (cos) AND imaginary (sin) parts.
        # The standard normal CF is real (phi_normal), so its imaginary part is 0.
        cos_emp = torch.cos(tz).mean(dim=0)  # (S, Q)
        sin_emp = torch.sin(tz).mean(dim=0)  # (S, Q)

        # |phi_emp - phi_normal|^2 = (cos_emp - phi_normal)^2 + sin_emp^2.
        # The sin^2 term makes the test sensitive to asymmetry/skew (and non-zero
        # mean); dropping it left the test blind to the odd moments.
        err = (cos_emp - self.phi_normal.unsqueeze(0)) ** 2 + sin_emp**2  # (S, Q)
        # Epps-Pulley statistic is N * integral; the N factor makes the statistic's
        # null distribution sample-size-independent (and keeps the gradient scale
        # consistent across batch sizes). Average over slices.
        n_samples = z.shape[0]
        per_slice = (self.weights.unsqueeze(0) * err).sum(dim=-1)  # (S,)
        return n_samples * per_slice.mean()


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
        self.n_slices = n_slices
        # NOTE: embed_dim is retained for API compatibility but no longer used to
        # pre-allocate directions — slices are drawn dynamically in forward().

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) — batch of embeddings.

        Returns:
            Scalar loss.
        """
        # NOTE: embeddings are fed to the test RAW (no centering/standardization).
        # SIGReg's whole purpose is to push the distribution to N(0, I); normalizing
        # mean/variance here would erase exactly the signal the loss must penalize and
        # remove the gradient that prevents representation collapse.
        #
        # Draw FRESH random unit projection directions every forward pass. With a
        # single fixed set of slices the encoder can make exactly those directions
        # Gaussian while leaving the rest of the sphere non-Gaussian; resampling makes
        # the sliced test cover the sphere over the course of training.
        directions = torch.randn(z.shape[-1], self.n_slices, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=0)
        projections = z @ directions  # (B, n_slices)

        return self.test(projections)


class SIGReg(nn.Module):
    """SIGReg loss combining invariance and Gaussianity regularization.

    Args:
        embed_dim: Dimension of the embeddings.
        n_slices: Number of random projection directions.
        t_max: Max quadrature point for CF test.
        n_quad: Number of quadrature points.
        sigreg_lambda: The single leJEPA hyperparameter lambda. The total loss is
            the convex combination
                lambda * SIGReg + (1 - lambda) * invariance.
            The paper/reference recommend a small value (~0.02).
    """

    def __init__(
        self,
        embed_dim: int,
        n_slices: int = 64,
        t_max: float = 3.0,
        n_quad: int = 17,
        sigreg_lambda: float = 0.02,
    ):
        super().__init__()
        self.sigreg_lambda = sigreg_lambda
        self.slicing_test = SlicingUnivariateTest(embed_dim, n_slices, t_max, n_quad)

    def forward(
        self, *views: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            *views: each (B, D) — embeddings from V augmented views of the same
                images. At least two views are required.

        Returns:
            (total_loss, metrics_dict)
        """
        assert len(views) >= 2, "SIGReg needs at least two views"
        # (V, B, D)
        z = torch.stack(views, dim=0)

        # Invariance: variance of the views about their per-sample mean, on RAW
        # embeddings. The reference does NOT L2-normalize: normalizing projects
        # embeddings onto the unit hypersphere (a SimCLR/cosine geometry), which is
        # inconsistent with matching an isotropic Gaussian in R^D and biases the
        # model toward collapse. For two views this equals (1/4)||z1 - z2||^2.
        inv_loss = (z.mean(dim=0, keepdim=True) - z).square().mean()

        # Regularization: average Gaussianity test over all views.
        reg_loss = sum(self.slicing_test(v) for v in views) / len(views)

        # leJEPA single-hyperparameter convex combination.
        total = self.sigreg_lambda * reg_loss + (1.0 - self.sigreg_lambda) * inv_loss

        metrics = {
            "invariance_loss": inv_loss.item(),
            "regularization_loss": reg_loss.item(),
            "total_loss": total.item(),
        }
        return total, metrics
