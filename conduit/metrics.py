from enum import Enum
from functools import partial
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, Union

from ranzen import str_to_enum
import torch
from torch import Tensor

__all__ = [
    "Comparator",
    "accuracy",
    "accuracy_per_subclass",
    "hard_prediction",
    "per_subclass_metric",
    "precision_at_k",
    "robust_accuracy",
    "subclass_balanced_accuracy",
    "tnr_per_subclass",
    "tpr_per_subclass",
]


@torch.no_grad()
def hard_prediction(logits: Tensor) -> Tensor:
    logits = torch.atleast_2d(logits.squeeze())
    return (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)


@torch.no_grad()
def accuracy(logits: Tensor, *, targets: Tensor) -> Tensor:
    logits = torch.atleast_2d(logits.squeeze())
    if len(logits) != len(targets):
        raise ValueError("'logits' and 'targets' must match in size at dimension 0.")
    preds = hard_prediction(logits)
    return (preds == targets).float().mean()


@torch.no_grad()
def precision_at_k(
    logits: Tensor, targets: Tensor, top_k: Union[int, Tuple[int, ...]] = (1,)
) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if isinstance(top_k, int):
        top_k = (top_k,)
    maxk = max(top_k)
    batch_size = targets.size(0)
    _, pred = logits.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res: List[Tensor] = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Comparator(Protocol):
    def __call__(
        self, y_pred: Tensor, *, y_true: Tensor
    ) -> Tuple[Tensor, Optional[Union[Tensor, slice]]]:
        """
        :param y_pred: Predicted labels or raw logits of a classifier.
        :param y_true: Ground truth (correct) labels.

        :returns: An element-wise comparison between ``y_pred`` and ``y_true`` or a subset of them;
        if the latter, the second element returned should be a mask indicating which samples
        comprise that subset.
        """
        ...


class Aggregator(Enum):
    MIN = min
    "Aggregate by taking the minimum."
    MAX = max
    "Aggregate by taking the maximum."
    MEAN = torch.mean
    "Aggregate by taking the mean."
    MEADIAN = torch.median
    "Aggregate by taking the median."


C = TypeVar("C", bound=Comparator, covariant=True)


class per_subclass_metric(Generic[C]):
    def __init__(self, comparator: C, aggregator: Optional[Union[Aggregator, str]] = None) -> None:
        """
        Computes a subclass-wise metric determined by a given comparator function.

        :param comparator: Function used to assess the correctness of ``y_pred`` with respect
        to ``y_true``.

        :param aggregator: Function with which to aggregate over the per-subclass scores.
        If ``None`` then no aggregation will be applied and scores will be returned for each
        subclass.
        """
        self.comparator = comparator
        if aggregator is not None:
            aggregator = str_to_enum(str_=aggregator, enum=Aggregator)
        self.aggregator = aggregator

    @torch.no_grad()
    def __call__(
        self,
        y_pred: Tensor,
        *,
        y_true: Tensor,
        s: Tensor,
    ) -> Tensor:
        """
        Compute the subclass-wise score(s) using the given comparator function.

        :param y_pred: Predicted labels or raw logits of a classifier.
        :param y_true: Ground truth (correct) labels.
        :param s: Ground turth labels indicating the subclass-membership of each sample.

        :returns: The score(s) as determined by the :attr:`comparator` and :attr:`aggregator`.

        :raises ValueError: If ``y_pred``, ``y_true``, and ``s`` do not match in size at dimension 0
        ('the batch dimension').

        """
        if len(y_pred) != len(y_true) != len(s):
            raise ValueError("'y_pred', 'y_true', and 's' must match in size at dimension 0.")
        # Interpret floating point predictions as potential logits and attempt to convert
        # them to hard predictions.
        if y_pred.is_floating_point():
            y_pred = hard_prediction(y_pred)

        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        s = s.squeeze()

        comps, mask = self.comparator(y_pred=y_pred, y_true=y_true)
        mask = slice(None) if mask is None else mask
        s_unique, s_counts = s.unique(return_counts=True)
        s_m = s[mask].flatten()[:, None] == s_unique[None]
        scores = (comps[:, None] * s_m).sum(0) / s_counts
        if self.aggregator is not None:
            return self.aggregator.value(scores)
        return scores


def equal(y_pred: Tensor, *, y_true: Tensor) -> Tuple[Tensor, None]:
    y_true = torch.atleast_1d(y_true.squeeze()).long()
    if len(y_pred) != len(y_true):
        raise ValueError("'y_pred' and 'y_true' must match in size at dimension 0.")
    if y_pred.is_floating_point():
        y_pred = hard_prediction(y_pred)
    return (y_pred == y_true).float(), None


def conditional_equal(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    y_pred_cond: Optional[int] = None,
    y_true_cond: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    mask = torch.ones_like(y_pred, dtype=torch.bool)
    if y_pred_cond is not None:
        mask &= y_pred == y_pred_cond
    if y_true_cond is not None:
        mask &= y_true == y_true_cond
    comps, _ = equal(y_pred=y_pred[mask], y_true=y_true[mask])
    return comps, mask


robust_accuracy = per_subclass_metric(comparator=equal, aggregator=Aggregator.MIN)
subclass_balanced_accuracy = per_subclass_metric(comparator=equal, aggregator=Aggregator.MEAN)
accuracy_per_subclass = per_subclass_metric(comparator=equal, aggregator=None)
tpr_per_subclass = per_subclass_metric(
    comparator=partial(conditional_equal, y_true_cond=1), aggregator=None
)
tnr_per_subclass = per_subclass_metric(
    comparator=partial(conditional_equal, y_true_cond=0), aggregator=None
)
