"""Hydra entry point for training. Dispatches to the appropriate trainer by mode.

Usage:
    python -m jepajitfusion.train_app --config-name=train_lejepa
    python -m jepajitfusion.train_app --config-name=train_jit
    python -m jepajitfusion.train_app --config-name=train_fusion
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.trainers.fusion_trainer import FusionTrainer
from jepajitfusion.trainers.jit_trainer import JiTTrainer
from jepajitfusion.trainers.lejepa_trainer import LeJEPATrainer


@hydra.main(version_base=None, config_path="conf", config_name="train_lejepa")
def train_app(cfg: DictConfig) -> None:
    # Hydra changes cwd; restore original
    os.chdir(hydra.utils.get_original_cwd())

    print(f"Training mode: {cfg.mode}")
    print(OmegaConf.to_yaml(cfg))

    match cfg.mode:
        case "lejepa":
            trainer = LeJEPATrainer(cfg)
        case "jit":
            trainer = JiTTrainer(cfg)
        case "fusion":
            trainer = FusionTrainer(cfg)
        case _:
            raise ValueError(f"Unknown training mode: {cfg.mode}")

    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    train_app()
