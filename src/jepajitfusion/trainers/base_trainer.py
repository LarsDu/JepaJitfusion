"""Abstract base trainer with common infrastructure."""

import glob
import os
import re
import uuid
from abc import ABC, abstractmethod

import torch

from jepajitfusion.trainers.summary import TrainingSummary
from jepajitfusion.utils import get_amp_dtype, get_device, set_seed


def generate_run_id(prefix: str) -> str:
    """Generate a unique run ID like 'jit_a1b2c3d4'."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class BaseTrainer(ABC):
    """Base class for all trainers.

    Handles: device selection, seed, AMP dtype, checkpoint I/O,
    and training summary tracking. Each run gets a unique directory
    under checkpoint_dir to prevent overwriting.
    """

    def __init__(self, seed: int, amp_dtype: str, checkpoint_dir: str, run_id: str):
        self.device = get_device()
        set_seed(seed)
        self.amp_dtype = get_amp_dtype(amp_dtype)
        self.summary = TrainingSummary()
        self.run_id = run_id
        self.checkpoint_dir = os.path.join(checkpoint_dir, run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Run: {self.run_id}")
        print(f"Output: {self.checkpoint_dir}")

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

    def find_latest_checkpoint(self, prefix: str) -> str | None:
        """Find the latest epoch checkpoint in the run directory.

        Looks for files matching '{prefix}_epoch_*.pth' and returns
        the one with the highest epoch number, or '{prefix}_last.pth'
        if it exists.
        """
        last_path = os.path.join(self.checkpoint_dir, f"{prefix}_last.pth")
        if os.path.exists(last_path):
            return last_path

        pattern = os.path.join(self.checkpoint_dir, f"{prefix}_epoch_*.pth")
        candidates = glob.glob(pattern)
        if not candidates:
            return None

        def _epoch_num(path: str) -> int:
            m = re.search(r"_epoch_(\d+)\.pth$", path)
            return int(m.group(1)) if m else -1

        return max(candidates, key=_epoch_num)
