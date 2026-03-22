"""Typed config for sampling.

Model architecture is loaded from the checkpoint — only sampling
parameters need to be specified here.
"""

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
    class_label: Optional[int] = None
