"""Hydra entry point for JiT diffusion decoder training.

Usage:
    python -m jepajitfusion.train_jit_app
    python -m jepajitfusion.train_jit_app num_epochs=200 batch_size=128
"""

import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import JiTTrainConfig
from jepajitfusion.trainers.jit_trainer import JiTTrainer

cs = ConfigStore.instance()
cs.store(name="train_jit", node=JiTTrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train_jit")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    config: JiTTrainConfig = OmegaConf.to_object(cfg)
    trainer = JiTTrainer(config)
    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
