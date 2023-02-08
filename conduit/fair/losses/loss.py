from torch import Tensor, nn
from typing_extensions import override

from conduit.types import Loss

__all__ = ["OnlineReweightingLoss"]


class OnlineReweightingLoss(nn.Module):
    """Wrapper that computes a loss balanced by intersectional group size."""

    def __init__(self, loss_fn: Loss) -> None:
        super().__init__()
        # the base loss function needs to produce instance-wise losses for the
        # reweighting (determined by subgroup cardinality) to be applied
        loss_fn.reduction = "none"
        self.loss_fn = loss_fn

    @override
    def forward(self, logits: Tensor, targets: Tensor, subgroup_inf: Tensor) -> Tensor:
        unweighted_loss = self.loss_fn(logits, targets)
        for _y in targets.unique():
            for _s in subgroup_inf.unique():
                # compute the cardinality of each subgroup and use this to weight the sample-losses
                mask = (targets == _y) & (subgroup_inf == _s)
                unweighted_loss[mask] /= mask.sum()
        return unweighted_loss.sum()
