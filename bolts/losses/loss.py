from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["CrossEntropy"]


class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(weight=weight, size_average=size_average, reduce=reduce, reduction="none")
        self.ignore_index = ignore_index
        self._reduction_str = reduction

    def forward(self, input: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        _target = target.view(-1).long()
        _weight = weight.view(-1) if weight is not None else torch.ones_like(_target)
        losses = F.cross_entropy(
            input,
            _target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        losses *= _weight
        if self._reduction_str == "mean":
            return losses.mean()
        if self._reduction_str == "none":
            return losses
        if self._reduction_str == "sum":
            return losses.sum()
