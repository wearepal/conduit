from abc import abstractmethod
from enum import Enum
from functools import partial, wraps
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
from torch import Tensor

__all__ = [
    "Comparator",
    "accuracy",
    "accuracy_per_class",
    "accuracy_per_group",
    "accuracy_per_subclass",
    "balanced_accuracy",
    "fscore",
    "fscore_per_class",
    "fscore_per_group",
    "fscore_per_subclass",
    "groupwise_metric",
    "hard_prediction",
    "macro_fscore",
    "max_difference_1d",
    "merge_indices",
    "nanmax",
    "nanmin",
    "pdist_1d",
    "precision_at_k",
    "robust_accuracy",
    "robust_fscore",
    "robust_fscore_gap",
    "robust_gap",
    "robust_tnr",
    "robust_tpr",
    "subclass_balanced_accuracy",
    "subclasswise_metric",
    "tnr_per_subclass",
    "tpr_differences",
    "tpr_per_subclass",
    "weighted_nanmean",
]


@torch.no_grad()
def hard_prediction(logits: Tensor) -> Tensor:
    logits = logits.squeeze(1) if logits.ndim == 2 else torch.atleast_1d(logits)
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
    @abstractmethod
    def __call__(self, y_pred: Tensor, *, y_true: Tensor) -> R:
        """Compare.

        :param y_pred: Predicted labels or raw logits of a classifier.
        :param y_true: Ground truth (correct) labels.

        :returns: An element-wise comparison between ``y_pred`` and ``y_true`` or a subset of them;
            if the latter, the second element returned should be a mask indicating which samples
            comprise that subset.
        """
        ...


C = TypeVar("C", bound=Comparator)
C_co = TypeVar("C_co", bound=Comparator, covariant=True)


@torch.no_grad()
def nanmax(x: Tensor) -> Tensor:
    return torch.amax(torch.nan_to_num(x, nan=-torch.inf))


@torch.no_grad()
def nanmin(x: Tensor) -> Tensor:
    return torch.amin(torch.nan_to_num(x, nan=torch.inf))


@torch.no_grad()
def pdist_1d(x: Tensor) -> Tensor:
    return (x - x.unsqueeze(-1)).abs().squeeze()


@torch.no_grad()
def max_difference_1d(x: Tensor) -> Tensor:
    return nanmax(pdist_1d(x))


@torch.no_grad()
def weighted_nanmean(x: Tensor, *, weights: Tensor) -> Tensor:
    if x.numel() != weights.numel():
        raise RuntimeError(
            "'weights' and the input tensor being weighted, 'x', must match in size."
        )
    if torch.any(weights < 0):
        raise RuntimeError("'weights' must contain only non-negative elements ")
    denom = weights.nansum()
    return (weights * x).nansum() / denom


class Aggregator(Enum):
    MIN = (nanmin,)
    "Aggregate by taking the minimum."
    MAX = (nanmax,)
    "Aggregate by taking the maximum."
    MEAN = (torch.nanmean,)
    "Aggregate by taking the mean."
    MEADIAN = (torch.nanmedian,)
    "Aggregate by taking the median."
    DIFF = (pdist_1d,)
    "Aggregate by taking the pairwise (absolute) differences."
    MAX_DIFF = (max_difference_1d,)
    "Aggregate by taking the maximum of the pairwise (absolute) differences."

    def __init__(self, fn: Callable[[Tensor], Tensor]) -> None:
        """
        Metric aggregator.

        :param fn: Aggregation function."""
        self.fn = fn

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Apply the aggregation function associated with the enum member to the input.

        :param x: Input to be aggregated.
        :returns: Aggregated input.
        """
        return self.fn(x)


@overload
def merge_indices(
    *indices: Tensor, return_cardinalities: Literal[True]
) -> Tuple[Tensor, List[int]]:
    ...


@overload
def merge_indices(*indices: Tensor, return_cardinalities: Literal[False] = ...) -> Tensor:
    ...


def merge_indices(
    *indices: Tensor, return_cardinalities: bool = False
) -> Union[Tensor, Tuple[Tensor, List[int]]]:
    """
    Bijectively merge a sequence of index tensors into a single index tensor, such that each
    combination of possible indices from across the elements in ``group_ids`` is assigned a unique
    index.

    :param indices: (Long-type) index tensors.
    :param return_cardinalities: if ``True``, return sizes.
    :returns: A merged index set which uniquely indexes each possible combination in indices.

    :raises TypeError: If any elemnts of ``indices`` do not have dtype ``torch.long``.
    """
    cards: Optional[List[int]] = None
    if return_cardinalities:
        cards = []
    group_ids_ls = list(indices)
    first_elem = group_ids_ls.pop().clone().squeeze()
    if first_elem.dtype != torch.long:
        raise TypeError("All index tensors must have dtype `torch.long'.")
    unique_vals, unique_inv = first_elem.unique(return_inverse=True)
    index_set = unique_inv
    if cards is not None:
        cards.append(len(unique_vals))

    for elem in group_ids_ls:
        elem = elem.squeeze()
        if elem.dtype != torch.long:
            raise TypeError("All index tensors must have dtype `torch.long'.")
        unique_vals, inv_indices = elem.unique(return_inverse=True)
        card = int(len(unique_vals))
        if cards is not None:
            cards.append(card)
        index_set *= card
        index_set += cast(Tensor, inv_indices)

    if cards is None:
        return index_set
    return index_set, cards


def nans(*sizes: int, device: Optional[torch.device] = None) -> Tensor:
    return torch.full(tuple(sizes), fill_value=torch.nan, device=device)


def nans_like(tensor: Tensor, *, device: Optional[torch.device] = None) -> Tensor:
    return torch.full_like(tensor, fill_value=torch.nan, device=device)


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
        If ``None`` then no aggregation will be applied and scores will be returned for each group.

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
        first_elem = group_ids_ls.pop().clone().squeeze()
        index_set = first_elem.unique(return_inverse=True)[1]

        for elem in group_ids_ls:
            if len(y_pred) != len(y_true) != len(elem):
                raise ValueError(
                    "'y_pred', 'y_true', and all elements of 'group_ids' must match in size at"
                    " dimension 0."
                )
            elem = elem.squeeze()
            unique_vals, inv_indices = elem.unique(return_inverse=True)
            index_set *= len(unique_vals)
            index_set += cast(Tensor, inv_indices)

    res = comparator(y_pred=y_pred, y_true=y_true)
    if isinstance(res, tuple):
        comps, comp_mask = res
    else:
        comps, comp_mask = res, slice(None)

    if index_set is not None:
        res = index_set.max()
        scores = torch.scatter_reduce(
            input=nans(int(index_set.max() + 1)),
            src=comps,
            dim=0,
            index=index_set[comp_mask],
            reduce="mean",
            include_self=False,
        )
        if aggregator is not None:
            scores = aggregator(scores)
        return scores

    return comps.mean()


A = TypeVar("A", Aggregator, None)
A_co = TypeVar("A_co", Aggregator, None, covariant=True)


class Metric(Protocol[C_co, A_co]):
    @staticmethod
    def __call__(y_pred: Tensor, *, y_true: Tensor) -> Tensor:
        ...


class GroupwiseMetric(Protocol[C_co, A_co]):
    @staticmethod
    def __call__(y_pred: Tensor, *, y_true: Tensor, s: Tensor) -> Tensor:
        ...


def groupwise_metric(
    comparator: C, *, aggregator: A, cond_on_pred: bool = False
) -> GroupwiseMetric[C, A]:
    """Converts a given ``comparator`` and ``aggregator`` into a group-wise metric.

    :param comparator: Function used to assess the correctness of ``y_pred`` with respect
        to ``y_true``. Should return a score for each sample.
    :param aggregator: Function with which to aggregate over the group-wise scores.
        If ``None`` then no aggregation will be applied and scores will be returned for each group.
    :param cond_on_pred: Whethr to condition on predictions.

    :returns: A group-wise metric formed from ``comparator`` and ``aggregator``.
    """

    @wraps(comparator)
    def _decorated_comparator(y_pred: Tensor, *, y_true: Tensor, s: Tensor) -> Tensor:
        return _apply_groupwise_metric(
            s,
            y_pred if cond_on_pred else y_true,
            comparator=comparator,
            aggregator=aggregator,
            y_pred=y_pred,
            y_true=y_true,
        )

    return _decorated_comparator


def subclasswise_metric(
    comparator: C,
    *,
    aggregator: A,
) -> GroupwiseMetric[C, A]:
    """Converts a given ``comparator`` and ``aggregator`` into a subclass-wise metric.

    :param comparator: Function used to assess the correctness of ``y_pred`` with respect
        to ``y_true``. Should return a score for each sample.

    :param aggregator: Function with which to aggregate over the subclass-wise scores.
        If ``None`` then no aggregation will be applied and scores will be returned for each group.

    :returns: A subclass-wise metric formed from ``comparator`` and ``aggregator``.
    """

    @wraps(comparator)
    def _decorated_comparator(y_pred: Tensor, *, y_true: Tensor, s: Tensor) -> Tensor:
        return _apply_groupwise_metric(
            s, comparator=comparator, aggregator=aggregator, y_pred=y_pred, y_true=y_true
        )

    return _decorated_comparator


def classwise_metric(
    comparator: C,
    *,
    aggregator: A,
    cond_on_pred: bool = False,
) -> Metric[C, A]:
    """Converts a given ``comparator`` and ``aggregator`` into a subclass-wise metric.

    :param comparator: Function used to assess the correctness of ``y_pred`` with respect
        to ``y_true``. Should return a score for each sample.

    :param aggregator: Function with which to aggregate over the subclass-wise scores.
        If ``None`` then no aggregation will be applied and scores will be returned for each group.
    :param cond_on_pred: Whethr to condition on predictions.

    :returns: A subclass-wise metric formed from ``comparator`` and ``aggregator``.
    """

    @wraps(comparator)
    def _decorated_comparator(y_pred: Tensor, *, y_true: Tensor) -> Tensor:
        return _apply_groupwise_metric(
            y_pred if cond_on_pred else y_true,
            comparator=comparator,
            aggregator=aggregator,
            y_pred=y_pred,
            y_true=y_true,
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
accuracy_per_subclass = subclasswise_metric(comparator=equal, aggregator=None)
subclass_balanced_accuracy = subclasswise_metric(comparator=equal, aggregator=Aggregator.MEAN)
robust_gap = subclasswise_metric(comparator=equal, aggregator=Aggregator.MAX_DIFF)

group_balanced_accuracy = groupwise_metric(comparator=equal, aggregator=Aggregator.MEAN)
accuracy_per_group = groupwise_metric(comparator=equal, aggregator=None)

accuracy_per_class = classwise_metric(comparator=equal, aggregator=None)
balanced_accuracy = classwise_metric(
    comparator=equal, aggregator=Aggregator.MEAN, cond_on_pred=False
)
precision_per_class = classwise_metric(comparator=equal, aggregator=None, cond_on_pred=True)
precision_per_subclass = groupwise_metric(comparator=equal, aggregator=None, cond_on_pred=True)

balanced_precision = classwise_metric(comparator=equal, aggregator=None, cond_on_pred=True)

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
robust_tpr = subclasswise_metric(
    comparator=partial(conditional_equal, y_true_cond=1), aggregator=Aggregator.MIN
)
robust_tnr = subclasswise_metric(
    comparator=partial(conditional_equal, y_true_cond=0), aggregator=Aggregator.MIN
)


@torch.no_grad()
def pad_to_size(size: int, *, src: Tensor, index: Tensor, value: float = 0.0) -> Tensor:
    padding = src.new_full((size,), fill_value=value)
    return padding.scatter_(dim=0, src=src, index=index)


def fscore(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    s: Optional[Tensor] = None,
    beta: float = 1.0,
    inner_summand: Optional[Literal["y_true", "s"]] = None,
    ignore_unsupported: bool = True,
    aggregator: Optional[Aggregator] = None,
) -> Tensor:
    """Computes F-beta score between ``y_pred`` and ``y_true`` with optional subclass-conditioning.

    :param y_pred: Predicted labels.
    :param y_true: Target labels.
    :param s: Subclass labels.
    :param beta: Beta coefficient that determines the weight of recall relative to precision.
        ``beta < 1`` lends more weight to precision, while ``beta > 1`` favors recal
    :param inner_summand: Which conditioning factor, if any, to sum over prior to the final
        aggregation when conditioning on the subclass labels.
    :param ignore_unsupported: Whether to ignore ``y_pred`` values that are unsupported by
        ``y_true`` (labels appearing in the predicted but not ground-truth labels). Note that
        class:`sklearn.metrics.f1_score` does *not* do this and instead includes unsupported values
        in its aggregation, i.e. classes included only in ``y_pred`` are included in the
        class-conditional F-scores as 0 components that influence the aggregation step.
    :param aggregator: Function with which to aggregate over the scores.
        If ``None`` then no aggregation will be applied and scores will be returned for each group.

    :returns: The (optionally aggregated) F-beta score given predictions ``y_pred`` and targets
        ``y_pred``.
    """
    # map the predicted and ground-truth labels to a common basis
    y_unique, rel_indices = torch.unique(torch.cat((y_true, y_pred)), return_inverse=True)
    y_true, y_pred = rel_indices.chunk(2)
    card_y = len(y_unique)

    if s is None:
        rec_ids = y_true
        rec_ids_u = rec_ids.unique()
        card_s = None
        prec_ids = y_pred
        target_size = card_y
    else:
        s_u, s_inv = s.unique(return_inverse=True)
        card_s = len(s_u)
        prec_ids = y_pred * card_s + s_inv
        rec_ids = y_true * card_s + s_inv
        rec_ids_u = rec_ids.unique()
        target_size = card_y * card_s

    precs = _apply_groupwise_metric(
        prec_ids,
        comparator=equal,
        y_pred=y_pred,
        y_true=y_true,
        aggregator=None,
    )
    recs = _apply_groupwise_metric(
        rec_ids,
        comparator=equal,
        y_pred=y_pred,
        y_true=y_true,
        aggregator=None,
    )

    if len(precs) < target_size:
        precs = pad_to_size(size=target_size, index=prec_ids.unique(), src=precs, value=torch.nan)
    if len(recs) < target_size:
        recs = pad_to_size(size=target_size, index=rec_ids_u, src=recs, value=torch.nan)

    beta_sq = beta**2
    f1s = (1 + beta_sq) * (precs * recs) / (beta_sq * precs + recs)
    if ignore_unsupported:
        # Ignore predictions that are unsupported by the ground-truth label set, supp(``y_true``).
        f1s = f1s.gather(0, rec_ids_u)
    f1s = f1s.nan_to_num_(nan=0.0)

    if (inner_summand is not None) and (s is not None):
        card_s = len(torch.unique(s))
        reduction_dim = int(inner_summand == "s")
        f1s = f1s.view(-1, card_s).nanmean(reduction_dim)

    if aggregator is None:
        return f1s
    return aggregator(f1s)


def robust_fscore(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    s: Tensor,
    beta: float = 1.0,
    ignore_unsupported: bool = True,
) -> Tensor:
    return fscore(
        y_pred=y_pred,
        y_true=y_true,
        s=s,
        beta=beta,
        inner_summand="y_true",
        ignore_unsupported=ignore_unsupported,
        aggregator=Aggregator.MIN,
    )


def fscore_per_class(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    beta: float = 1.0,
    ignore_unsupported: bool = True,
) -> Tensor:
    return fscore(
        y_pred=y_pred,
        y_true=y_true,
        beta=beta,
        inner_summand=None,
        ignore_unsupported=ignore_unsupported,
        aggregator=None,
    )


def fscore_per_subclass(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    s: Tensor,
    beta: float = 1.0,
    ignore_unsupported: bool = True,
) -> Tensor:
    return fscore(
        y_pred=y_pred,
        y_true=y_true,
        s=s,
        beta=beta,
        inner_summand="y_true",
        ignore_unsupported=ignore_unsupported,
        aggregator=None,
    )


def macro_fscore(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    beta: float = 1.0,
    ignore_unsupported: bool = True,
) -> Tensor:
    return fscore(
        y_pred=y_pred,
        y_true=y_true,
        s=None,
        beta=beta,
        inner_summand=None,
        ignore_unsupported=ignore_unsupported,
        aggregator=Aggregator.MEAN,
    )


def fscore_per_group(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    s: Tensor,
    beta: float = 1.0,
    ignore_unsupported: bool = True,
) -> Tensor:
    return fscore(
        y_pred=y_pred,
        y_true=y_true,
        s=s,
        beta=beta,
        inner_summand=None,
        ignore_unsupported=ignore_unsupported,
        aggregator=None,
    )


def robust_fscore_gap(
    y_pred: Tensor,
    *,
    y_true: Tensor,
    s: Tensor,
    beta: float = 1.0,
    ignore_unsupported: bool = True,
) -> Tensor:
    return fscore(
        y_pred=y_pred,
        y_true=y_true,
        s=s,
        beta=beta,
        inner_summand="y_true",
        ignore_unsupported=ignore_unsupported,
        aggregator=Aggregator.MAX_DIFF,
    )
