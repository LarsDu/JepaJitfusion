"""Abstract base trainer with common infrastructure."""

import os
from abc import ABC, abstractmethod

import torch

from jepajitfusion.trainers.summary import TrainingSummary
from jepajitfusion.utils import get_amp_dtype, get_device, set_seed


class BaseTrainer(ABC):
    """Base class for all trainers.

    Handles: device selection, seed, AMP dtype, checkpoint I/O,
    and training summary tracking.
    """

    def __init__(self, seed: int, amp_dtype: str, checkpoint_dir: str):
        self.device = get_device()
        set_seed(seed)
        self.amp_dtype = get_amp_dtype(amp_dtype)
        self.summary = TrainingSummary()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    @abstractmethod
    def train(self, train_loader, val_loader=None) -> TrainingSummary:
        """Run the training loop. Must be implemented by subclasses."""

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save a checkpoint with arbitrary data."""
        torch.save(kwargs, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint."""
        return torch.load(path, map_location="cpu", weights_only=False)
