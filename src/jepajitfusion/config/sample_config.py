from dataclasses import dataclass
from typing import Optional


@dataclass
class SampleConfig:
    checkpoint_path: str = "checkpoints/jit_last.pth"
    output_dir: str = "samples"
    num_samples: int = 16
    num_steps: int = 50
    cfg_scale: float = 1.5
    seed: int = 1999
    conditioning_mode: str = "none"
    encoder_checkpoint: str = ""
    class_label: Optional[int] = None
    img_size: int = 64
    num_channels: int = 3
