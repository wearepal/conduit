from enum import Enum
from functools import partial, wraps
from typing import List, Optional, Protocol, Tuple, TypeVar, Union

import torch
from torch import Tensor

__all__ = [
    "Comparator",
    "accuracy",
    "accuracy_per_subclass",
    "groupwise_metric",
    "hard_prediction",
    "max_difference_1d",
    "precision_at_k",
    "robust_accuracy",
    "robust_gap",
    "subclass_balanced_accuracy",
    "subclasswise_metric",
    "tnr_per_subclass",
    "tpr_differences",
    "tpr_per_subclass",
]


@torch.no_grad()
def hard_prediction(logits: Tensor) -> Tensor:
    logits = torch.atleast_1d(logits.squeeze())
    return (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)


@torch.no_grad()
def accuracy(y_pred: Tensor, *, y_true: Tensor) -> Tensor:
    y_pred = torch.atleast_1d(y_pred.squeeze())
    y_true = torch.atleast_1d(y_true.squeeze())
    if len(y_pred) != len(y_true):
        raise ValueError("'logits' and 'targets' must match in size at dimension 0.")
    preds = hard_prediction(y_pred)
    return (preds == y_true).float().mean()


@torch.no_grad()
def precision_at_k(
    y_pred: Tensor, *, y_true: Tensor, top_k: Union[int, Tuple[int, ...]] = (1,)
) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if isinstance(top_k, int):
        top_k = (top_k,)
    maxk = max(top_k)
    batch_size = y_true.size(0)
    _, pred = y_pred.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    res: List[Tensor] = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


R = TypeVar("R", Tensor, Tuple[Tensor, Union[Tensor, slice]], covariant=True)


class Comparator(Protocol[R]):
    def __call__(self, y_pred: Tensor, *, y_true: Tensor) -> R:
        """
        :param y_pred: Predicted labels or raw logits of a classifier.
        :param y_true: Ground truth (correct) labels.

        :returns: An element-wise comparison between ``y_pred`` and ``y_true`` or a subset of them;
        if the latter, the second element returned should be a mask indicating which samples
        comprise that subset.
        """
        ...


C = TypeVar("C", bound=Comparator)
C_co = TypeVar("C_co", bound=Comparator, covariant=True)


def _pdist_1d(x: Tensor) -> Tensor:
    return torch.pdist(x.view(-1, 1)).squeeze()


def max_difference_1d(x: Tensor) -> Tensor:
    return _pdist_1d(x).max()


class Aggregator(Enum):
    MIN = min
    "Aggregate by taking the minimum."
    MAX = max
    "Aggregate by taking the maximum."
    MEAN = torch.mean
    "Aggregate by taking the mean."
    MEADIAN = torch.median
    "Aggregate by taking the median."
    DIFF = partial(_pdist_1d)
    "Aggregate by taking the pairwise (absolute) differences."
    MAX_DIFF = partial(max_difference_1d)
    "Aggregate by taking the maximum of the pairwise (absolute) differences."


@torch.no_grad()
def _apply_groupwise_metric(
    *group_ids: Tensor,
    comparator: Comparator,
    aggregator: Optional[Aggregator],
    y_pred: Tensor,
    y_true: Tensor,
) -> Tensor:
    """
    Computes a groupwise metric given a ``comparator`` and ``aggregator``.

    :param comparator: Function used to assess the correctness of ``y_pred`` with respect
    to ``y_true``. Should return a score for each sample.

    :param aggregator: Function with which to aggregate over the group-wise scores.
    If ``None`` then no aggregation will be applied and scores will be returned for each
    group.

    :param y_pred: Predictions to be scored
    :param y_true: Ground truth (correct) labels.
    :param group_ids: Ground truth labels indicating the group-membership of each sample.

    :returns: The score(s) as determined by the :attr:`comparator` and :attr:`aggregator`.

    :raises ValueError: If ``y_pred``, ``y_true``, and ``s`` do not match in size at dimension 0
    (the 'batch' dimension).

    """
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    index_set = None

    if group_ids:
        group_ids_ls = list(group_ids)
        index_set = group_ids_ls.pop().squeeze()

        for elem in group_ids_ls:
            if len(y_pred) != len(y_true) != len(elem):
                raise ValueError(
                    "'y_pred', 'y_true', and all elements of 'group_ids' must match in size at dimension 0."
                )
            elem = elem.squeeze()
            unique_vals, inv_indices = elem.unique(return_inverse=True)
            index_set *= len(unique_vals)
            index_set += inv_indices

    res = comparator(y_pred=y_pred, y_true=y_true)
    if isinstance(res, tuple):
        comps, comp_mask = res
    else:
        comps, comp_mask = res, slice(None)

    if index_set is not None:
        scores = torch.scatter_reduce(comps, dim=0, index=index_set[comp_mask], reduce="mean")
        if aggregator is not None:
            scores = aggregator.value(scores)
        return scores

    return comps.mean()


A = TypeVar("A", Aggregator, None)
A_co = TypeVar("A_co", Aggregator, None, covariant=True)


class GroupwiseMetric(Protocol[C_co, A_co]):
    @staticmethod
    def __call__(y_pred: Tensor, *, y_true: Tensor, s: Tensor) -> Tensor:
        ...


def groupwise_metric(
    comparator: C,
    *,
    aggregator: A,
) -> GroupwiseMetric[C, A]:
    """
    Converts a given ``comparator`` and ``aggregator`` into a group-wise metric.

    :param comparator: Function used to assess the correctness of ``y_pred`` with respect
    to ``y_true``. Should return a score for each sample.

    :param aggregator: Function with which to aggregate over the group-wise scores.
    If ``None`` then no aggregation will be applied and scores will be returned for each
    group.

    :returns: A group-wise metric formed from ``comparator`` and ``aggregator``.
    """

    @wraps(comparator)
    def _decorated_comparator(y_pred: Tensor, *, y_true: Tensor, s: Tensor) -> Tensor:
        return _apply_groupwise_metric(
            s, y_true, comparator=comparator, aggregator=aggregator, y_pred=y_pred, y_true=y_true
        )

    return _decorated_comparator


def subclasswise_metric(
    comparator: C,
    *,
    aggregator: A,
) -> GroupwiseMetric[C, A]:
    """
    Converts a given ``comparator`` and ``aggregator`` into a subclass-wise metric.

    :param comparator: Function used to assess the correctness of ``y_pred`` with respect
    to ``y_true``. Should return a score for each sample.

    :param aggregator: Function with which to aggregate over the subclass-wise scores.
    If ``None`` then no aggregation will be applied and scores will be returned for each
    group.

    :returns: A subclass-wise metric formed from ``comparator`` and ``aggregator``.
    """

    @wraps(comparator)
    def _decorated_comparator(y_pred: Tensor, *, y_true: Tensor, s: Tensor) -> Tensor:
        return _apply_groupwise_metric(
            s, comparator=comparator, aggregator=aggregator, y_pred=y_pred, y_true=y_true
        )

    return _decorated_comparator


def equal(y_pred: Tensor, *, y_true: Tensor) -> Tensor:
    y_true = torch.atleast_1d(y_true.squeeze()).long()
    if len(y_pred) != len(y_true):
        raise ValueError("'y_pred' and 'y_true' must match in size at dimension 0.")
    # Interpret floating point predictions as potential logits and attempt to convert
    # them to hard predictions.
    if y_pred.is_floating_point():
        y_pred = hard_prediction(y_pred)
    return (y_pred == y_true).float()


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
    comps = equal(y_pred=y_pred[mask], y_true=y_true[mask])
    return comps, mask


robust_accuracy = subclasswise_metric(comparator=equal, aggregator=Aggregator.MIN)
robust_gap = subclasswise_metric(comparator=equal, aggregator=Aggregator.MAX_DIFF)
subclass_balanced_accuracy = subclasswise_metric(comparator=equal, aggregator=Aggregator.MEAN)
group_balanced_accuracy = groupwise_metric(comparator=equal, aggregator=Aggregator.MEAN)
accuracy_per_subclass = subclasswise_metric(comparator=equal, aggregator=None)
tpr_per_subclass = subclasswise_metric(
    comparator=partial(conditional_equal, y_true_cond=1), aggregator=None
)
tnr_per_subclass = subclasswise_metric(
    comparator=partial(conditional_equal, y_true_cond=0), aggregator=None
)
tpr_differences = subclasswise_metric(
    comparator=partial(conditional_equal, y_true_cond=1), aggregator=Aggregator.DIFF
)
tnr_differences = subclasswise_metric(
    comparator=partial(conditional_equal, y_true_cond=0), aggregator=Aggregator.DIFF
)
