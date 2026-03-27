"""Typed training configs — one per pipeline.

Each config contains only the fields relevant to its pipeline.
Sub-configs (DataConfig, EncoderConfig, DecoderConfig) are composed as typed fields.
"""

from dataclasses import dataclass, field

from jepajitfusion.config.data_config import DataConfig
from jepajitfusion.config.decoder_config import DecoderConfig
from jepajitfusion.config.encoder_config import EncoderConfig


@dataclass
class LeJEPATrainConfig:
    """Config for self-supervised LeJEPA encoder training."""

    # Sub-configs
    dataset: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    # Training
    num_epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    beta1: float = 0.9
    beta2: float = 0.95
    ema_decays: list[float] = field(default_factory=lambda: [0.9999])
    amp_dtype: str = "bfloat16"
    seed: int = 1999
    checkpoint_dir: str = "checkpoints"
    run_id: str = ""
    log_every: int = 10
    sample_every: int = 50
    validate_every: int = 10
    # Multi-crop
    n_global_crops: int = 2
    n_local_crops: int = 4
    global_crop_size: int = 96
    local_crop_size: int = 32
    global_crop_scale_min: float = 0.3
    global_crop_scale_max: float = 1.0
    local_crop_scale_min: float = 0.05
    local_crop_scale_max: float = 0.3
    # SIGReg
    sigreg_n_slices: int = 64
    sigreg_t_max: float = 3.0
    sigreg_n_quad: int = 17
    invariance_weight: float = 25.0
    regularization_weight: float = 1.0


@dataclass
class JiTTrainConfig:
    """Config for JiT diffusion decoder training."""

    # Sub-configs
    dataset: DataConfig = field(default_factory=DataConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # Training
    num_epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 20
    beta1: float = 0.9
    beta2: float = 0.95
    ema_decays: list[float] = field(default_factory=lambda: [0.9996, 0.9998, 0.9999])
    amp_dtype: str = "bfloat16"
    seed: int = 1999
    checkpoint_dir: str = "checkpoints"
    run_id: str = ""
    log_every: int = 10
    sample_every: int = 50
    validate_every: int = 10
    # Diffusion
    P_mean: float = -0.8
    P_std: float = 0.8
    noise_scale: float = 0.375
    cfg_scale: float = 1.5


@dataclass
class FusionTrainConfig:
    """Config for fusion training: frozen encoder -> conditioned decoder."""

    # Sub-configs
    dataset: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # Training
    num_epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 20
    beta1: float = 0.9
    beta2: float = 0.95
    ema_decays: list[float] = field(default_factory=lambda: [0.9996, 0.9998, 0.9999])
    amp_dtype: str = "bfloat16"
    seed: int = 1999
    checkpoint_dir: str = "checkpoints"
    run_id: str = ""
    log_every: int = 10
    sample_every: int = 50
    validate_every: int = 10
    # Diffusion
    P_mean: float = -0.8
    P_std: float = 0.8
    noise_scale: float = 0.375
    cfg_scale: float = 1.5
    # Fusion-specific
    encoder_checkpoint: str = "checkpoints/lejepa_last.pth"
    freeze_encoder: bool = True
