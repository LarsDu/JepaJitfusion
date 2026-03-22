"""Training summary for tracking losses across epochs."""

from dataclasses import dataclass, field


@dataclass
class TrainingSummary:
    """Tracks training metrics across epochs."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    def add_train_loss(self, loss: float) -> None:
        self.train_losses.append(loss)

    def add_val_loss(self, loss: float) -> None:
        self.val_losses.append(loss)
