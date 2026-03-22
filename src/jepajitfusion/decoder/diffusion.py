"""Diffusion utilities for JiT training.

Implements:
- Logit-normal time sampling (JiT Algorithm 1)
- z_t linear interpolation
- v-loss computation (x-prediction → velocity conversion)
"""

import torch
import torch.nn.functional as F


def sample_logit_normal_time(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample timesteps from logit-normal distribution.

    t = sigmoid(Normal(P_mean, P_std))

    Args:
        batch_size: Number of timesteps to sample.
        P_mean: Mean of the logit-normal distribution.
        P_std: Std of the logit-normal distribution.
        device: Device for the output tensor.

    Returns:
        (B,) timesteps in (0, 1).
    """
    u = torch.randn(batch_size, device=device) * P_std + P_mean
    return torch.sigmoid(u)


def compute_z_t(
    x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation between noise and clean image.

    z_t = t * x + (1 - t) * noise

    Args:
        x: (B, C, H, W) clean images.
        noise: (B, C, H, W) noise (already scaled by noise_scale).
        t: (B,) timesteps.

    Returns:
        (B, C, H, W) noisy images.
    """
    t = t.view(-1, 1, 1, 1)

    # See: hhttps://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/denoiser.py#L55 # noqa: E501
    return t * x + (1 - t) * noise


def compute_v_loss(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    conditioning=None,
    noise_scale: float = 0.25,
) -> torch.Tensor:
    """Compute v-loss for JiT training (Algorithm 1).

    The model predicts clean images (x-prediction). We convert to velocity
    and compute MSE against the target velocity.

    Args:
        model: JiT model that takes (z_t, t, conditioning) → x_pred.
        x: (B, C, H, W) clean images in [-1, 1].
        t: (B,) timesteps in (0, 1).
        noise: (B, C, H, W) standard Gaussian noise.
        conditioning: Optional conditioning input.
        noise_scale: Noise scaling factor (img_size / 256).

    Returns:
        Scalar v-loss.
    """
    # Scale noise
    scaled_noise = noise * noise_scale

    # Interpolate to get noisy image
    z_t = compute_z_t(x, scaled_noise, t)

    # Model predicts clean image
    x_pred = model(z_t, t, conditioning)

    # Convert to velocity and compute loss
    ## See: https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/denoiser.py#L56 noqa: E501
    t_view = t.view(-1, 1, 1, 1)
    denom = (1 - t_view).clamp(min=0.05)
    v_pred = (x_pred - z_t) / denom
    v_target = (x - z_t) / denom

    return F.mse_loss(v_pred, v_target)
