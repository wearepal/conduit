"""Accuracy that accepts sens label."""
import torch
from torchmetrics import Accuracy


class FbAccuracy(Accuracy):
    """Accuracy where sens can be passed."""

    @property
    def __name__(self) -> str:
        return f"Accuracy"

    def update(self, preds: torch.Tensor, sens: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            sens: Ground truth sensitive labels
            target: Ground truth values
        """
        super().update(preds, target)
