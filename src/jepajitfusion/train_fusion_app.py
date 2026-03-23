"""Hydra entry point for fusion training (frozen encoder -> conditioned decoder).

Usage:
    python -m jepajitfusion.train_fusion_app
    python -m jepajitfusion.train_fusion_app encoder_checkpoint=checkpoints/lejepa_last.pth
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import DataConfig, DecoderConfig, EncoderConfig, FusionTrainConfig
from jepajitfusion.trainers.fusion_trainer import FusionTrainer


@hydra.main(version_base=None, config_path="conf", config_name="train_fusion")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    raw = OmegaConf.to_container(cfg, resolve=True)
    config = FusionTrainConfig(
        dataset=DataConfig(**raw.pop("dataset")),
        encoder=EncoderConfig(**raw.pop("encoder")),
        decoder=DecoderConfig(**raw.pop("decoder")),
        **raw,
    )

    trainer = FusionTrainer(config)
    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
