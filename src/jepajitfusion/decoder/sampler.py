"""Heun ODE solver for JiT sampling (Algorithm 2)."""

import torch


class HeunSampler:
    """Heun (2nd-order) ODE solver for flow-matching sampling.

    Supports classifier-free guidance (CFG) when both conditional
    and unconditional inputs are provided.

    The sampling process goes from t=0 (noise) to t=1 (clean image).
    """

    def __init__(
        self,
        num_steps: int = 50,
        cfg_scale: float = 1.5,
        noise_scale: float = 0.25,
    ):
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.noise_scale = noise_scale

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple[int, ...],
        device: torch.device,
        conditioning=None,
        uncond_conditioning=None,
    ) -> torch.Tensor:
        """Generate samples using Heun's method.

        Args:
            model: JiT model.
            shape: (B, C, H, W) output shape.
            device: Device.
            conditioning: Optional conditioning for guided sampling.
            uncond_conditioning: Unconditional conditioning for CFG.

        Returns:
            (B, C, H, W) generated images in [-1, 1].
        """
        # Start from noise at t=0
        z = torch.randn(shape, device=device) * self.noise_scale

        for i in range(self.num_steps):
            t_val = i / self.num_steps
            t_next_val = (i + 1) / self.num_steps
            dt = t_next_val - t_val

            t = torch.full((shape[0],), t_val, device=device)
            t_next = torch.full((shape[0],), t_next_val, device=device)

            # Velocity at current point
            v = self._get_velocity(model, z, t, conditioning, uncond_conditioning)

            # Euler step
            z_euler = z + dt * v

            # Heun correction (skip at last step)
            if i < self.num_steps - 1:
                v_next = self._get_velocity(
                    model, z_euler, t_next, conditioning, uncond_conditioning
                )
                v_avg = 0.5 * (v + v_next)
                z = z + dt * v_avg
            else:
                z = z_euler

        return z

    def _get_velocity(
        self, model, z, t, cond, uncond_cond
    ) -> torch.Tensor:
        """Compute velocity, optionally with classifier-free guidance."""
        x_pred = model(z, t, cond)
        t_view = t.view(-1, 1, 1, 1)
        denom = (1 - t_view).clamp(min=0.05)
        v = (x_pred - z) / denom

        if self.cfg_scale > 1.0 and uncond_cond is not None:
            x_pred_uncond = model(z, t, uncond_cond)
            v_uncond = (x_pred_uncond - z) / denom
            v = v_uncond + self.cfg_scale * (v - v_uncond)

        return v
