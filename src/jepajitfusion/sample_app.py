"""Hydra entry point for sampling from a trained JiT model.

Model architecture is loaded from the checkpoint (decoder_config + dataset_config).

Usage:
    python -m jepajitfusion.sample_app
    python -m jepajitfusion.sample_app checkpoint_path=checkpoints/jit_last.pth num_samples=32
"""

import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import DataConfig, DecoderConfig, SampleConfig
from jepajitfusion.data.transforms import reverse_transform
from jepajitfusion.decoder.jit_model import JiTModel
from jepajitfusion.decoder.sampler import HeunSampler
from jepajitfusion.utils import get_device, set_seed

cs = ConfigStore.instance()
cs.store(name="sample", node=SampleConfig)


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def sample_app(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())

    config: SampleConfig = OmegaConf.to_object(cfg)
    device = get_device()
    set_seed(config.seed)

    # Load checkpoint
    print(f"Loading checkpoint from {config.checkpoint_path}...")
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    if "decoder_config" not in checkpoint or "dataset_config" not in checkpoint:
        raise RuntimeError(
            "Checkpoint missing 'decoder_config' or 'dataset_config'. "
            "Re-train with the updated JiTTrainer/FusionTrainer to embed typed configs."
        )

    dec = DecoderConfig(**checkpoint["decoder_config"])
    ds = DataConfig(**checkpoint["dataset_config"])

    # Build model from checkpoint configs
    model = JiTModel(
        img_size=ds.img_size,
        patch_size=dec.patch_size,
        in_channels=ds.num_channels,
        dim=dec.dim,
        depth=dec.depth,
        num_heads=dec.num_heads,
        mlp_ratio=dec.mlp_ratio,
        bottleneck_dim=dec.bottleneck_dim,
        num_classes=dec.num_classes,
        conditioning_mode=dec.conditioning_mode,
        jepa_dim=dec.jepa_dim,
    )

    # Try EMA model first, fall back to base model
    if "ema_state_dicts" in checkpoint and checkpoint["ema_state_dicts"]:
        model.load_state_dict(checkpoint["ema_state_dicts"][0])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights")

    model = model.to(device)
    model.eval()

    # Sample
    sampler = HeunSampler(
        num_steps=config.num_steps,
        cfg_scale=config.cfg_scale,
        noise_scale=ds.img_size / 256.0,
    )

    print(f"Generating {config.num_samples} samples...")
    shape = (config.num_samples, ds.num_channels, ds.img_size, ds.img_size)

    conditioning = None
    if dec.conditioning_mode == "label" and config.class_label is not None:
        conditioning = torch.full(
            (config.num_samples,), config.class_label, dtype=torch.long, device=device
        )

    samples = sampler.sample(model, shape, device, conditioning=conditioning)

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    rev_t = reverse_transform()
    for i, s in enumerate(samples):
        img = rev_t(s)
        img.save(os.path.join(config.output_dir, f"sample_{i:04d}.png"))

    print(f"Saved {config.num_samples} samples to {config.output_dir}")


if __name__ == "__main__":
    sample_app()
