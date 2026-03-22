from dataclasses import dataclass


@dataclass
class DataConfig:
    name: str = "pokemon_64"
    img_size: int = 64
    num_channels: int = 3
    test_size: float = 0.15
    data_dir: str = "downloads"
    num_classes: int = 0
