import torch

from conduit.metrics import (
    accuracy_per_subclass,
    robust_accuracy,
    robust_gap,
    tpr_differences,
    tpr_per_subclass,
)


def test_groupwise_metrics() -> None:
    y_true = torch.randint(0, 2, (50, 1))
    y_pred = torch.randint(0, 2, (50, 1))
    s = torch.randint(0, 5, (50, 1))

    card_y = len(y_true.unique())
    card_s = len(s.unique())

    res = accuracy_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(res) == card_s
    res = tpr_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(res) == card_s
    res = robust_accuracy(y_pred=y_pred, y_true=y_true, s=s)
    assert res.ndim == 0
    res = robust_gap(y_pred=y_pred, y_true=y_true, s=s)
    assert res.ndim == 0
    res = tpr_differences(y_pred=y_pred, y_true=y_true, s=s)
    assert len(res) == (card_y * card_s)
