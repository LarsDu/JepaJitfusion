"""LeJEPA trainer: self-supervised encoder training with SIGReg loss."""

import os

import torch
from omegaconf import DictConfig

from jepajitfusion.data.datasets import MultiCropDataset, get_dataloader, get_dataset, multicrop_collate
from jepajitfusion.encoder.multicrop import MultiCropAugmentation
from jepajitfusion.encoder.projection_head import ProjectionHead
from jepajitfusion.encoder.sigreg import SIGReg
from jepajitfusion.encoder.vit import VisionTransformer
from jepajitfusion.models.ema import MultiEMA
from jepajitfusion.trainers.base_trainer import BaseTrainer
from jepajitfusion.trainers.summary import TrainingSummary
from jepajitfusion.utils import get_cosine_schedule_with_warmup


class LeJEPATrainer(BaseTrainer):
    """Self-supervised encoder training with multi-crop augmentation and SIGReg loss.

    Training procedure:
    1. Generate multiple augmented crops of each image
    2. Pass global crops through encoder + projection head
    3. Compute SIGReg loss (invariance + Gaussianity regularization)
    4. Update encoder via gradient descent, update EMA
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        enc = cfg.encoder
        ds = cfg.dataset

        # Build encoder
        self.encoder = VisionTransformer(
            img_size=ds.img_size,
            patch_size=enc.patch_size,
            in_channels=ds.num_channels,
            embed_dim=enc.embed_dim,
            depth=enc.depth,
            num_heads=enc.num_heads,
            mlp_ratio=enc.mlp_ratio,
        ).to(self.device)

        # Projection head: embed_dim → 2*embed_dim → embed_dim
        self.projector = ProjectionHead(
            enc.embed_dim, enc.embed_dim * 2, enc.embed_dim
        ).to(self.device)

        # SIGReg loss
        self.sigreg = SIGReg(
            embed_dim=enc.embed_dim,
            n_slices=cfg.sigreg_n_slices,
            t_max=cfg.sigreg_t_max,
            n_quad=cfg.sigreg_n_quad,
            invariance_weight=cfg.invariance_weight,
            regularization_weight=cfg.regularization_weight,
        ).to(self.device)

        # EMA
        self.ema = MultiEMA(self.encoder, list(cfg.ema_decays))

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )

        # Multi-crop augmentation
        self.multicrop = MultiCropAugmentation(
            n_global=cfg.n_global_crops,
            n_local=cfg.n_local_crops,
            global_size=cfg.global_crop_size,
            local_size=cfg.local_crop_size,
            global_scale=(cfg.global_crop_scale_min, cfg.global_crop_scale_max),
            local_scale=(cfg.local_crop_scale_min, cfg.local_crop_scale_max),
        )

        n_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        print(f"LeJEPA encoder: {n_params:.1f}M parameters")

    def _build_dataloaders(self):
        """Build train/test dataloaders with multi-crop augmentation."""
        ds = self.cfg.dataset
        # Get raw datasets (no transform — multicrop handles it)
        train_ds, test_ds = get_dataset(
            ds.name, transform=None, data_dir=ds.data_dir, test_size=ds.test_size
        )
        # Wrap train set with multi-crop
        train_ds = MultiCropDataset(train_ds, self.multicrop)
        train_loader = get_dataloader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=multicrop_collate,
        )
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
            self.encoder.train()
            self.projector.train()
            epoch_loss = 0.0

            for batch_idx, (crops, _labels) in enumerate(train_loader):
                # crops: list of tensors, first n_global are global crops
                global_crops = [c.to(self.device) for c in crops[: self.multicrop.n_global]]

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.amp_dtype
                ):
                    # Encode both global crops
                    z1 = self.projector(self.encoder(global_crops[0]))
                    z2 = self.projector(self.encoder(global_crops[1]))

                    loss, metrics = self.sigreg(z1, z2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.ema.update(self.encoder)

                epoch_loss += loss.item()

                if batch_idx % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  [{epoch}/{self.cfg.num_epochs}][{batch_idx}/{len(train_loader)}] "
                        f"loss={loss.item():.4f} inv={metrics['invariance_loss']:.4f} "
                        f"reg={metrics['regularization_loss']:.4f} lr={lr:.2e}"
                    )

            avg_loss = epoch_loss / len(train_loader)
            self.summary.add_train_loss(avg_loss)
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % self.cfg.sample_every == 0 or epoch == self.cfg.num_epochs - 1:
                self.save_checkpoint(
                    os.path.join(self.cfg.checkpoint_dir, f"lejepa_epoch_{epoch + 1}.pth"),
                    epoch=epoch,
                    model_state_dict=self.encoder.state_dict(),
                    projector_state_dict=self.projector.state_dict(),
                    ema_state_dicts=[m.state_dict() for m in self.ema.ema_models],
                    optimizer_state_dict=self.optimizer.state_dict(),
                )

        # Save final checkpoint
        self.save_checkpoint(
            os.path.join(self.cfg.checkpoint_dir, "lejepa_last.pth"),
            epoch=self.cfg.num_epochs - 1,
            model_state_dict=self.encoder.state_dict(),
            projector_state_dict=self.projector.state_dict(),
            ema_state_dicts=[m.state_dict() for m in self.ema.ema_models],
            optimizer_state_dict=self.optimizer.state_dict(),
        )

        return self.summary
