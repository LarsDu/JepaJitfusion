"""JiT diffusion trainer: unconditional or class-conditioned image generation."""

import os

import torch
from omegaconf import DictConfig

from jepajitfusion.data.datasets import get_dataloader, get_dataset
from jepajitfusion.data.transforms import forward_transform, reverse_transform
from jepajitfusion.decoder.diffusion import compute_v_loss, sample_logit_normal_time
from jepajitfusion.decoder.jit_model import JiTModel
from jepajitfusion.decoder.sampler import HeunSampler
from jepajitfusion.models.ema import MultiEMA
from jepajitfusion.trainers.base_trainer import BaseTrainer
from jepajitfusion.trainers.summary import TrainingSummary
from jepajitfusion.utils import get_cosine_schedule_with_warmup


class JiTTrainer(BaseTrainer):
    """Trains a JiT diffusion model (unconditional or class-conditioned).

    Training uses flow-matching with logit-normal time sampling and v-loss.
    Periodically generates samples using the Heun ODE solver.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        dec = cfg.decoder
        ds = cfg.dataset

        self.model = JiTModel(
            img_size=ds.img_size,
            patch_size=dec.patch_size,
            in_channels=ds.num_channels,
            dim=dec.dim,
            depth=dec.depth,
            num_heads=dec.num_heads,
            mlp_ratio=dec.mlp_ratio,
            bottleneck_dim=dec.bottleneck_dim,
            num_classes=getattr(dec, "num_classes", 0),
            conditioning_mode=dec.conditioning_mode,
        ).to(self.device)

        self.ema = MultiEMA(self.model, list(cfg.ema_decays))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )

        self.noise_scale = cfg.noise_scale
        self.P_mean = cfg.P_mean
        self.P_std = cfg.P_std
        self.sampler = HeunSampler(
            num_steps=50, cfg_scale=cfg.cfg_scale, noise_scale=cfg.noise_scale
        )
        self.img_size = ds.img_size
        self.num_channels = ds.num_channels

    def _build_dataloaders(self):
        ds = self.cfg.dataset
        transform = forward_transform(ds.img_size)
        train_ds, test_ds = get_dataset(
            ds.name, transform=transform, data_dir=ds.data_dir, test_size=ds.test_size
        )
        train_loader = get_dataloader(train_ds, batch_size=self.cfg.batch_size)
        return train_loader

    def train(self, train_loader=None, val_loader=None) -> TrainingSummary:
        if train_loader is None:
            train_loader = self._build_dataloaders()

        total_steps = self.cfg.num_epochs * len(train_loader)
        warmup_steps = self.cfg.warmup_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                cond = None
                if self.model.conditioning_mode == "label":
                    cond = labels.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.amp_dtype
                ):
                    B = images.shape[0]
                    t = sample_logit_normal_time(
                        B, self.P_mean, self.P_std, device=self.device
                    )
                    noise = torch.randn_like(images)
                    loss = compute_v_loss(
                        self.model, images, t, noise, cond, self.noise_scale
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.ema.update(self.model)

                epoch_loss += loss.item()

                if batch_idx % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  [{epoch}/{self.cfg.num_epochs}][{batch_idx}/{len(train_loader)}] "
                        f"loss={loss.item():.4f} lr={lr:.2e}"
                    )

            avg_loss = epoch_loss / len(train_loader)
            self.summary.add_train_loss(avg_loss)
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            # Sample and checkpoint periodically
            if (epoch + 1) % self.cfg.sample_every == 0:
                self._sample_and_save(epoch)
                self._save_checkpoint(epoch)

        # Final save
        self._save_checkpoint(self.cfg.num_epochs - 1, suffix="last")
        return self.summary

    @torch.no_grad()
    def _sample_and_save(self, epoch: int, n_samples: int = 16) -> None:
        """Generate and save sample images using EMA model."""
        ema_model = self.ema.get_model(0)
        ema_model.eval()

        shape = (n_samples, self.num_channels, self.img_size, self.img_size)
        samples = self.sampler.sample(ema_model, shape, self.device)

        # Save as grid
        rev_t = reverse_transform()
        sample_dir = os.path.join(self.cfg.checkpoint_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for i, s in enumerate(samples):
            img = rev_t(s)
            img.save(os.path.join(sample_dir, f"epoch{epoch + 1}_sample{i}.png"))
        print(f"  Saved {n_samples} samples to {sample_dir}")

    def _save_checkpoint(self, epoch: int, suffix: str | None = None) -> None:
        name = f"jit_{suffix}.pth" if suffix else f"jit_epoch_{epoch + 1}.pth"
        self.save_checkpoint(
            os.path.join(self.cfg.checkpoint_dir, name),
            epoch=epoch,
            model_state_dict=self.model.state_dict(),
            ema_state_dicts=[m.state_dict() for m in self.ema.ema_models],
            optimizer_state_dict=self.optimizer.state_dict(),
            cfg={
                "img_size": self.img_size,
                "num_channels": self.num_channels,
                "conditioning_mode": self.model.conditioning_mode,
            },
        )
