"""Hydra entry point for JiT diffusion decoder training.

Usage:
    python -m jepajitfusion.train_jit_app
    python -m jepajitfusion.train_jit_app num_epochs=200 batch_size=128
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from jepajitfusion.config import DataConfig, DecoderConfig, JiTTrainConfig
from jepajitfusion.trainers.jit_trainer import JiTTrainer


@hydra.main(version_base=None, config_path="conf", config_name="train_jit")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))

    raw = OmegaConf.to_container(cfg, resolve=True)
    config = JiTTrainConfig(
        dataset=DataConfig(**raw.pop("dataset")),
        decoder=DecoderConfig(**raw.pop("decoder")),
        **raw,
    )

    trainer = JiTTrainer(config)
    summary = trainer.train()
    print(f"Training complete. Final loss: {summary.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
