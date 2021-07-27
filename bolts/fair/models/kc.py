"""K&C Baseline model."""
from kit import implements
from torch import Tensor

from bolts.data.structures import TernarySampleIW
from bolts.fair.models.erm import ErmBaseline

__all__ = ["KC"]


class KC(ErmBaseline):
    """Kamiran and Calders instance weighting method."""

    @implements(ErmBaseline)
    def _get_loss(self, logits: Tensor, *, batch: TernarySampleIW) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y.float(), instance_weight=batch.iw)
