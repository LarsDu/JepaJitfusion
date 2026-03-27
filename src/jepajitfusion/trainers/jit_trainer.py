"""JiT diffusion trainer: unconditional or class-conditioned image generation."""

import dataclasses
import os

import torch

from jepajitfusion.config import JiTTrainConfig
from jepajitfusion.data.datasets import get_dataloader, get_dataset
from jepajitfusion.data.transforms import eval_transform, forward_transform, reverse_transform
from jepajitfusion.decoder.diffusion import compute_v_loss, sample_logit_normal_time
from jepajitfusion.decoder.jit_model import JiTModel
from jepajitfusion.decoder.sampler import HeunSampler
from jepajitfusion.models.ema import MultiEMA
from jepajitfusion.trainers.base_trainer import BaseTrainer, generate_run_id
from jepajitfusion.trainers.summary import TrainingSummary
from jepajitfusion.utils import get_cosine_schedule_with_warmup


class JiTTrainer(BaseTrainer):
    """Trains a JiT diffusion model (unconditional or class-conditioned).

    Training uses flow-matching with logit-normal time sampling and v-loss.
    Periodically generates samples using the Heun ODE solver.
    """

    def __init__(self, config: JiTTrainConfig):
        run_id = config.run_id or generate_run_id("jit")
        super().__init__(
            seed=config.seed,
            amp_dtype=config.amp_dtype,
            checkpoint_dir=config.checkpoint_dir,
            run_id=run_id,
        )
        self.config = config
        dec = config.decoder
        ds = config.dataset

        self.model = JiTModel(
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
        ).to(self.device)

        self.ema = MultiEMA(self.model, config.ema_decays)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        self.sampler = HeunSampler(
            num_steps=50, cfg_scale=config.cfg_scale, noise_scale=config.noise_scale
        )

        # Resume from latest checkpoint if run_id was explicitly provided
        self.start_epoch = 0
        if config.run_id:
            self._try_resume()

    def _try_resume(self) -> None:
        """Attempt to resume from the latest checkpoint in the run directory."""
        ckpt_path = self.find_latest_checkpoint("jit")
        if ckpt_path is None:
            return

        ckpt = self.load_checkpoint(ckpt_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for ema_model, state in zip(self.ema.ema_models, ckpt["ema_state_dicts"]):
            ema_model.load_state_dict(state)
        self.start_epoch = ckpt["epoch"] + 1
        self.summary.train_losses = ckpt.get("train_losses", [])
        self.summary.val_losses = ckpt.get("val_losses", [])
        print(f"Resumed from {ckpt_path} (epoch {ckpt['epoch']})")

    def _build_dataloaders(self):
        ds = self.config.dataset
        transform = forward_transform(ds.img_size)
        val_tf = eval_transform(ds.img_size)
        train_ds, val_ds = get_dataset(
            ds.name,
            transform=transform,
            val_transform=val_tf,
            data_dir=ds.data_dir,
            test_size=ds.test_size,
        )
        train_loader = get_dataloader(train_ds, batch_size=self.config.batch_size)
        val_loader = get_dataloader(
            val_ds, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def train(self, train_loader=None, val_loader=None) -> TrainingSummary:
        if train_loader is None:
            train_loader, val_loader = self._build_dataloaders()

        total_steps = self.config.num_epochs * len(train_loader)
        warmup_steps = self.config.warmup_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Fast-forward scheduler if resuming
        if self.start_epoch > 0:
            skip_steps = self.start_epoch * len(train_loader)
            for _ in range(skip_steps):
                scheduler.step()
            print(f"Resuming from epoch {self.start_epoch}, scheduler advanced {skip_steps} steps")

        for epoch in range(self.start_epoch, self.config.num_epochs):
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
                        B, self.config.P_mean, self.config.P_std, device=self.device
                    )
                    noise = torch.randn_like(images)
                    loss = compute_v_loss(
                        self.model, images, t, noise, cond, self.config.noise_scale
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.ema.update(self.model)

                epoch_loss += loss.item()

                if batch_idx % self.config.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  [{epoch}/{self.config.num_epochs}][{batch_idx}/{len(train_loader)}] "
                        f"loss={loss.item():.4f} lr={lr:.2e}"
                    )

            avg_loss = epoch_loss / len(train_loader)
            self.summary.add_train_loss(avg_loss)
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            # Validation
            if val_loader is not None and (epoch + 1) % self.config.validate_every == 0:
                def _val_loss_fn(batch):
                    images, labels = batch
                    images = images.to(self.device)
                    cond = None
                    if self.model.conditioning_mode == "label":
                        cond = labels.to(self.device)
                    B = images.shape[0]
                    t = sample_logit_normal_time(
                        B, self.config.P_mean, self.config.P_std, device=self.device
                    )
                    noise = torch.randn_like(images)
                    return compute_v_loss(
                        self.model, images, t, noise, cond, self.config.noise_scale
                    ).item()

                self.model.eval()
                val_loss = self._validate_epoch(val_loader, _val_loss_fn)
                print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

            # Sample and checkpoint periodically
            if (epoch + 1) % self.config.sample_every == 0:
                self._sample_and_save(epoch)
                self._save_checkpoint(epoch)

        # Final save
        self._save_checkpoint(self.config.num_epochs - 1, suffix="last")
        return self.summary

    @torch.no_grad()
    def _sample_and_save(self, epoch: int, n_samples: int = 16) -> None:
        """Generate and save sample images using EMA model."""
        ema_model = self.ema.get_model(0)
        ema_model.eval()

        ds = self.config.dataset
        dec = self.config.decoder
        shape = (n_samples, ds.num_channels, ds.img_size, ds.img_size)

        cond = None
        uncond_cond = None
        if self.model.conditioning_mode == "label":
            cond = torch.randint(0, dec.num_classes, (n_samples,), device=self.device)
            uncond_cond = torch.full(
                (n_samples,), dec.num_classes, device=self.device, dtype=torch.long
            )

        samples = self.sampler.sample(
            ema_model, shape, self.device, conditioning=cond, uncond_conditioning=uncond_cond
        )

        # Save as grid
        rev_t = reverse_transform()
        sample_dir = os.path.join(self.checkpoint_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for i, s in enumerate(samples):
            img = rev_t(s)
            img.save(os.path.join(sample_dir, f"epoch{epoch + 1}_sample{i}.png"))
        print(f"  Saved {n_samples} samples to {sample_dir}")

    def _save_checkpoint(self, epoch: int, suffix: str | None = None) -> None:
        name = f"jit_{suffix}.pth" if suffix else f"jit_epoch_{epoch + 1}.pth"
        self.save_checkpoint(
            os.path.join(self.checkpoint_dir, name),
            epoch=epoch,
            run_id=self.run_id,
            model_state_dict=self.model.state_dict(),
            ema_state_dicts=[m.state_dict() for m in self.ema.ema_models],
            optimizer_state_dict=self.optimizer.state_dict(),
            decoder_config=dataclasses.asdict(self.config.decoder),
            dataset_config=dataclasses.asdict(self.config.dataset),
            train_losses=self.summary.train_losses,
            val_losses=self.summary.val_losses,
        )
