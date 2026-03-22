from dataclasses import dataclass


@dataclass
class EncoderConfig:
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    patch_size: int = 8
