import math
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_dtype(dtype_str: str = "bfloat16") -> torch.dtype:
    """Convert string dtype name to torch dtype for AMP."""
    match dtype_str:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case _:
            return torch.float32


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
