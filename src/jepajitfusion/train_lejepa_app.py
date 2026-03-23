"""Hydra entry point for LeJEPA self-supervised encoder training.

Usage:
    python -m jepajitfusion.train_lejepa_app
    python -m jepajitfusion.train_lejepa_app num_epochs=100 learning_rate=1e-4
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import DataConfig, EncoderConfig, LeJEPATrainConfig
from jepajitfusion.trainers.lejepa_trainer import LeJEPATrainer


@hydra.main(version_base=None, config_path="conf", config_name="train_lejepa")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    raw = OmegaConf.to_container(cfg, resolve=True)
    config = LeJEPATrainConfig(
        dataset=DataConfig(**raw.pop("dataset")),
        encoder=EncoderConfig(**raw.pop("encoder")),
        **raw,
    )

    trainer = LeJEPATrainer(config)
    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
