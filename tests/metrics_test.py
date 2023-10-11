"""Test metrics."""
from typing import Final

import numpy as np
import pytest
from sklearn.metrics import f1_score
import torch
from torch import Tensor

import conduit.metrics as cdtm

NUM_SAMPLES: Final[int] = 212


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator().manual_seed(47)


def assert_close(a: Tensor | np.ndarray | float, b: Tensor | np.ndarray | float, /) -> None:
    if isinstance(a, (np.ndarray, float, int)):
        a = torch.as_tensor(a, dtype=torch.float32)
    if isinstance(b, (np.ndarray, float, int)):
        b = torch.as_tensor(b, dtype=torch.float32)
    a = torch.nan_to_num(a, nan=torch.inf)
    b = torch.nan_to_num(b, nan=torch.inf)
    torch.testing.assert_close(a, b)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("card_y", [1, 2, 3])
@pytest.mark.parametrize("card_s", [1, 2, 3])
@pytest.mark.parametrize("disjoint", [True, False])
def test_groupwise_metrics(
    generator: torch.Generator, card_y: int, card_s: int, disjoint: bool
) -> None:
    y_true = torch.randint(0, card_y, (NUM_SAMPLES, 1), generator=generator)
    y_pred = y_true.clone()
    # Peturb y_true by noise factor 0.3 to simulate predictions.
    m = torch.rand(NUM_SAMPLES, generator=generator) < 0.3
    y_pred[m] += 1
    y_pred[m] %= card_y
    # render the predicted and ground-truth labels disjoint
    if disjoint:
        y_pred += card_y

    s = torch.randint(0, card_s, (NUM_SAMPLES, 1))
    card_y = len(y_true.unique())
    card_s = len(s.unique())
    card_sy = card_y * card_s

    hits = (y_pred == y_true).float()
    acc = cdtm.accuracy(y_pred=y_pred, y_true=y_true)
    acc_ref = hits.mean()
    assert_close(acc, acc_ref)

    _, s_counts = s.unique(return_counts=True)
    sw_hits = cdtm.nans(NUM_SAMPLES, card_s).scatter_(-1, index=s, src=hits)

    aps = cdtm.accuracy_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(aps) == card_s
    aps_ref = sw_hits.nansum(0).div_(s_counts)
    assert_close(aps, aps_ref)

    sba = cdtm.subclass_balanced_accuracy(y_true=y_pred, y_pred=y_true, s=s)
    assert sba.ndim == 0
    sba_ref = aps_ref.nanmean()
    assert_close(sba, sba_ref)

    tps = cdtm.tpr_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(tps) == card_s
    y_mask = y_true == 1
    if y_mask.count_nonzero():
        tps_ref = sw_hits[y_mask.squeeze()].nansum(0)
        denom = s[y_mask].unique(return_counts=True)[1].float()
        if len(denom) < card_s:
            # pad
            denom = cdtm.nans_like(tps_ref).scatter_(0, index=s[y_mask].unique(), src=denom)
        tps_ref /= denom
    else:
        tps_ref = cdtm.nans(card_s)
    assert_close(tps_ref, tps)

    apc = cdtm.accuracy_per_class(y_pred=y_pred, y_true=y_true)
    _, y_counts = y_true.unique(return_counts=True)
    cw_hits = cdtm.nans(NUM_SAMPLES, card_y).scatter_(-1, index=y_true, src=hits)
    apc_ref = cw_hits.nansum(0).div_(y_counts)
    assert_close(apc_ref, apc)

    ra = cdtm.robust_accuracy(y_pred=y_pred, y_true=y_true, s=s)
    ra_ref = aps_ref.amin()
    assert ra.ndim == 0
    assert_close(ra_ref, ra)

    rg = cdtm.robust_gap(y_pred=y_pred, y_true=y_true, s=s)
    rg_ref = (aps - aps.unsqueeze(-1)).abs().amax()
    assert rg.ndim == 0
    assert_close(rg, rg_ref)

    td = cdtm.tpr_differences(y_pred=y_pred, y_true=y_true, s=s)
    td_ref = (tps - tps.unsqueeze(-1)).abs().squeeze()
    assert_close(td_ref, td)

    apg = cdtm.accuracy_per_group(y_pred=y_pred, y_true=y_true, s=s)
    assert len(apg) == card_sy

    ba_ref = apc_ref.mean()
    ba = cdtm.balanced_accuracy(y_pred=y_pred, y_true=y_true)
    assert_close(ba, ba_ref)

    fs = cdtm.macro_fscore(y_pred=y_pred, y_true=y_true)
    assert fs.ndim == 0
    fs_ref = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    assert_close(fs, fs_ref)  # type: ignore

    fpg = cdtm.fscore_per_group(y_pred=y_pred, y_true=y_true, s=s)
    assert len(fpg) == card_sy
    rfsg = cdtm.robust_fscore_gap(y_pred=y_pred, y_true=y_true, s=s)
    assert rfsg.ndim == 0
    fpc = cdtm.fscore_per_class(y_pred=y_pred, y_true=y_true)
    assert len(fpc) == card_y
    fps = cdtm.fscore_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(fps) == card_s
    fps_ref = torch.as_tensor(
        [
            f1_score(y_true=y_true[s == s_], y_pred=y_pred[s == s_], average="macro")
            for s_ in s.unique()
        ],
        dtype=torch.float32,
    )
    assert_close(fps, fps_ref)

    rfs = cdtm.robust_fscore(y_true=y_true, y_pred=y_pred, s=s)
    assert rfs.ndim == 0
    rfs_ref = fps_ref.amin()
    assert_close(rfs, rfs_ref)
