"""Test metrics."""
from typing import Final

import pytest
from sklearn.metrics import f1_score  # pyright: ignore
import torch

import conduit.metrics as cdtm

NUM_SAMPLES: Final[int] = 1000


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator().manual_seed(47)


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
    _, s_counts = s.unique(return_counts=True)
    sw_hits = torch.zeros(NUM_SAMPLES, card_s).scatter_(-1, index=s, src=hits)

    aps = cdtm.accuracy_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(aps) == card_s
    aps_ref = sw_hits.sum(0).div_(s_counts)
    torch.testing.assert_close(aps, aps_ref)  # pyright: ignore

    sba = cdtm.subclass_balanced_accuracy(y_true=y_pred, y_pred=y_true, s=s)
    assert sba.ndim == 0
    sba_ref = aps_ref.mean()
    torch.testing.assert_close(sba, sba_ref)  # pyright: ignore

    tps = cdtm.tpr_per_subclass(y_pred=y_pred, y_true=y_true, s=s)
    assert len(tps) == card_s
    y_mask = y_true == 1
    if y_mask.count_nonzero():
        tps_ref = sw_hits[y_mask.squeeze()].sum(0)
        denom = s[y_mask].unique(return_counts=True)[1].float()
        if len(denom) < card_s:
            # pad
            denom = torch.ones_like(tps_ref).scatter_(0, index=s[y_mask].unique(), src=denom)
        tps_ref /= denom
    else:
        tps_ref = torch.zeros(card_s)
    torch.testing.assert_close(tps_ref, tps)  # pyright: ignore

    ra = cdtm.robust_accuracy(y_pred=y_pred, y_true=y_true, s=s)
    assert ra.ndim == 0

    rg = cdtm.robust_gap(y_pred=y_pred, y_true=y_true, s=s)
    assert rg.ndim == 0

    td = cdtm.tpr_differences(y_pred=y_pred, y_true=y_true, s=s)
    td_ref = torch.pdist(tps.view(-1, 1)).squeeze()
    torch.testing.assert_close(td, td_ref)  # pyright: ignore

    apc = cdtm.accuracy_per_class(y_pred=y_pred, y_true=y_true)
    _, y_counts = y_true.unique(return_counts=True)
    cw_hits = torch.zeros(NUM_SAMPLES, card_y).scatter_(-1, index=y_true, src=hits)
    apc_ref = cw_hits.sum(0).div_(y_counts)
    torch.testing.assert_close(apc, apc_ref)  # pyright: ignore

    apg = cdtm.accuracy_per_group(y_pred=y_pred, y_true=y_true, s=s)
    assert len(apg) == card_sy

    ba_ref = apc_ref.mean()
    ba = cdtm.balanced_accuracy(y_pred=y_pred, y_true=y_true)
    torch.testing.assert_close(ba, ba_ref)  # pyright: ignore

    fs = cdtm.macro_fscore(y_pred=y_pred, y_true=y_true)
    assert fs.ndim == 0
    fs_ref = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    torch.testing.assert_close(fs.item(), fs_ref.item())  # pyright: ignore

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
    torch.testing.assert_close(fps, fps_ref)  # pyright: ignore

    rfs = cdtm.robust_fscore(y_true=y_true, y_pred=y_pred, s=s)
    assert rfs.ndim == 0
    rfs_ref = fps_ref.amin()
    torch.testing.assert_close(rfs, rfs_ref)  # pyright: ignore
