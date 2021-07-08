from typing import Optional

from kit import parsable
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing_extensions import Literal

__all__ = ["CrossEntropy", "OnlineReweightingLoss"]
ReductionType = Literal["mean", "none", "sum"]


class CrossEntropy(nn.CrossEntropyLoss):
    @parsable
    def __init__(
        self,
        class_weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__(weight=class_weight, reduction="none")
        self.ignore_index = ignore_index
        self._reduction_str = reduction

    def forward(
        self, input: Tensor, target: Tensor, instance_weight: Optional[Tensor] = None
    ) -> Tensor:
        _target = target.view(-1).long()
        losses = F.cross_entropy(
            input,
            _target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        if instance_weight is not None:
            _weight = instance_weight.view(-1)
            losses *= _weight
        if self._reduction_str == "mean":
            return losses.mean()
        if self._reduction_str == "none":
            return losses
        if self._reduction_str == "sum":
            return losses.sum()


class OnlineReweightingLoss(nn.Module):
    """Wrapper that computes a loss balanced by subgroups."""

    def __init__(self, loss_fn: _Loss) -> None:
        super().__init__()
        # the base loss function needs to produce instance-wise losses for the
        # reweighting (determined by subgroup cardinality) to be applied
        loss_fn.reduction = "none"
        self.loss_fn = loss_fn

    def forward(self, logits: Tensor, targets: Tensor, subgroup_inf: Tensor) -> Tensor:
        unweighted_loss = self.loss_fn(logits, targets)
        for _y in targets.unique():
            for _s in subgroup_inf.unique():
                # compute the cardinality of each subgroup and use this to weight the sample-losses
                mask = (targets == _y) & (subgroup_inf == _s)
                unweighted_loss[mask] /= mask.sum()
        return unweighted_loss.sum()
