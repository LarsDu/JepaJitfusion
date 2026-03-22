"""Multi-decay Exponential Moving Average for model parameters."""

import copy

import torch
from torch import nn


class MultiEMA(nn.Module):
    """Maintains multiple EMA copies of a model at different decay rates.

    Usage:
        ema = MultiEMA(model, decays=[0.9996, 0.9998, 0.9999])
        # After each optimizer step:
        ema.update(model)
        # For sampling/eval:
        ema_model = ema.get_model(idx=0)  # get the 0.9996 decay copy
    """

    def __init__(self, model: nn.Module, decays: list[float] | None = None):
        super().__init__()
        if decays is None:
            decays = [0.9999]
        self.decays = decays
        self.ema_models = nn.ModuleList(
            [self._copy_model(model) for _ in decays]
        )

    @staticmethod
    def _copy_model(model: nn.Module) -> nn.Module:
        ema = copy.deepcopy(model)
        ema.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_model, decay in zip(self.ema_models, self.decays):
            for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)

    def get_model(self, idx: int = 0) -> nn.Module:
        return self.ema_models[idx]
