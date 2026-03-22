from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainConfig:
    mode: str = "lejepa"  # "lejepa", "jit", "fusion"
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
    log_every: int = 10
    sample_every: int = 50
    # Diffusion params
    P_mean: float = -0.8
    P_std: float = 0.8
    noise_scale: float = 0.25
    cfg_scale: float = 1.5
    # Fusion params
    encoder_checkpoint: str = ""
    freeze_encoder: bool = True
    # LeJEPA multi-crop params
    n_global_crops: int = 2
    n_local_crops: int = 4
    global_crop_size: int = 64
    local_crop_size: int = 32
    global_crop_scale_min: float = 0.3
    global_crop_scale_max: float = 1.0
    local_crop_scale_min: float = 0.05
    local_crop_scale_max: float = 0.3
    # SIGReg params
    sigreg_n_slices: int = 64
    sigreg_t_max: float = 3.0
    sigreg_n_quad: int = 17
    invariance_weight: float = 25.0
    regularization_weight: float = 1.0
    # Hydra-populated sub-configs
    dataset: Any = None
    encoder: Any = None
    decoder: Any = None
