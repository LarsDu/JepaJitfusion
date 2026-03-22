"""Hydra entry point for LeJEPA self-supervised encoder training.

Usage:
    python -m jepajitfusion.train_lejepa_app
    python -m jepajitfusion.train_lejepa_app num_epochs=100 learning_rate=1e-4
"""

import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import LeJEPATrainConfig
from jepajitfusion.trainers.lejepa_trainer import LeJEPATrainer

cs = ConfigStore.instance()
cs.store(name="train_lejepa", node=LeJEPATrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train_lejepa")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    config: LeJEPATrainConfig = OmegaConf.to_object(cfg)
    trainer = LeJEPATrainer(config)
    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
