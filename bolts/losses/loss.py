from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

__all__ = ["CrossEntropy", "OnlineReweightingLoss"]


class CrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _target = target.view(-1).long()
        return super().forward(input, _target)


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
