"""Hydra entry point for fusion training (frozen encoder → conditioned decoder).

Usage:
    python -m jepajitfusion.train_fusion_app
    python -m jepajitfusion.train_fusion_app encoder_checkpoint=checkpoints/lejepa_last.pth
"""

import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import FusionTrainConfig
from jepajitfusion.trainers.fusion_trainer import FusionTrainer

cs = ConfigStore.instance()
cs.store(name="train_fusion", node=FusionTrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train_fusion")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    config: FusionTrainConfig = OmegaConf.to_object(cfg)
    trainer = FusionTrainer(config)
    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
