"""Hydra entry point for sampling from a trained JiT model.

Usage:
    python -m jepajitfusion.sample_app
    python -m jepajitfusion.sample_app checkpoint_path=checkpoints/jit_last.pth num_samples=32
"""

import os

import hydra
import torch
from omegaconf import DictConfig

from jepajitfusion.data.transforms import reverse_transform
from jepajitfusion.decoder.jit_model import JiTModel
from jepajitfusion.decoder.sampler import HeunSampler
from jepajitfusion.utils import get_device, set_seed


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def sample_app(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())

    device = get_device()
    set_seed(cfg.seed)

    # Load checkpoint
    print(f"Loading checkpoint from {cfg.checkpoint_path}...")
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("cfg", {})

    img_size = ckpt_cfg.get("img_size", cfg.img_size)
    num_channels = ckpt_cfg.get("num_channels", cfg.num_channels)
    conditioning_mode = ckpt_cfg.get("conditioning_mode", cfg.conditioning_mode)

    # Build model — we need to know the architecture params
    # For now, try loading the state dict into a default model
    # and let it fail if architecture mismatches
    model = JiTModel(
        img_size=img_size,
        in_channels=num_channels,
        conditioning_mode=conditioning_mode,
    )

    # Try EMA model first, fall back to base model
    if "ema_state_dicts" in checkpoint and checkpoint["ema_state_dicts"]:
        model.load_state_dict(checkpoint["ema_state_dicts"][0])
        print("Loaded EMA model weights")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights")

    model = model.to(device)
    model.eval()

    # Sample
    sampler = HeunSampler(
        num_steps=cfg.num_steps,
        cfg_scale=cfg.cfg_scale,
        noise_scale=img_size / 256.0,
    )

    print(f"Generating {cfg.num_samples} samples...")
    shape = (cfg.num_samples, num_channels, img_size, img_size)

    conditioning = None
    if conditioning_mode == "label" and cfg.class_label is not None:
        conditioning = torch.full(
            (cfg.num_samples,), cfg.class_label, dtype=torch.long, device=device
        )

    samples = sampler.sample(model, shape, device, conditioning=conditioning)

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    rev_t = reverse_transform()
    for i, s in enumerate(samples):
        img = rev_t(s)
        img.save(os.path.join(cfg.output_dir, f"sample_{i:04d}.png"))

    print(f"Saved {cfg.num_samples} samples to {cfg.output_dir}")


if __name__ == "__main__":
    sample_app()
