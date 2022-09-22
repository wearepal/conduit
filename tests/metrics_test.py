import torch

import conduit.metrics as cdtm


def test_groupwise_metrics() -> None:
    N = 57
    CARD_Y = 2
    CARD_S = 2

    y_true = torch.randint(0, CARD_Y, (N, 1))
    y_pred = y_true.clone()
    # Peturb y_true by noise factor 0.3 to simulate predictions.
    m = torch.rand(N) < 0.3
    y_pred[m] += 1
    y_pred[m] %= 2
    s = torch.randint(0, CARD_S, (N, 1))

    card_y = len(y_true.unique())
    card_s = len(s.unique())

    hits = (y_pred == y_true).float()
    _, s_counts = s.unique(return_counts=True)
    sw_hits = torch.zeros(N, CARD_S).scatter_(-1, index=s, src=hits)

    aps = cdtm.accuracy_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(aps) == card_s
    aps_ref = sw_hits.sum(0).div_(s_counts)
    assert torch.allclose(aps, aps_ref)

    sba = cdtm.subclass_balanced_accuracy(y_true=y_pred, y_pred=y_true, s=s)
    assert sba.ndim == 0
    ba_ref = aps_ref.mean()
    assert torch.allclose(sba, ba_ref)

    tps = cdtm.tpr_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(tps) == card_s
    y_mask = y_true == 1
    tps_ref = sw_hits[y_mask.squeeze()].sum(0) / s[y_mask].unique(return_counts=True)[1]
    assert torch.allclose(tps_ref, tps)

    ra = cdtm.robust_accuracy(y_pred=y_pred, y_true=y_true, s=s)
    assert ra.ndim == 0

    rg = cdtm.robust_gap(y_pred=y_pred, y_true=y_true, s=s)
    assert rg.ndim == 0

    td = cdtm.tpr_differences(y_pred=y_pred, y_true=y_true, s=s)
    td_ref = torch.pdist(tps.view(-1, 1)).squeeze()
    assert torch.allclose(td, td_ref)

    apc = cdtm.accuracy_per_class(y_pred=y_pred, y_true=y_true)
    _, y_counts = y_true.unique(return_counts=True)
    cw_hits = torch.zeros(N, CARD_Y).scatter_(-1, index=y_true, src=hits)
    apc_ref = cw_hits.sum(0).div_(y_counts)
    assert torch.allclose(apc, apc_ref)

    ba_ref = apc_ref.mean()
    ba = cdtm.balanced_accuracy(y_pred=y_pred, y_true=y_true)
    assert torch.allclose(ba, ba_ref)
