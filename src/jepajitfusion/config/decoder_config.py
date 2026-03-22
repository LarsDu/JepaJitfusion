from dataclasses import dataclass


@dataclass
class DecoderConfig:
    dim: int = 384
    depth: int = 8
    num_heads: int = 6
    patch_size: int = 4
    bottleneck_dim: int = 64
    mlp_ratio: float = 4.0
    num_classes: int = 0
    conditioning_mode: str = "none"  # "none", "label", "jepa"
    jepa_dim: int = 256
