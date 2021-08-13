from __future__ import annotations
from pathlib import Path

import pytest
import torch
from torchvision import transforms as T
from torchvision.datasets import VisionDataset

from bolts.data import (
    BinarySample,
    BinarySampleIW,
    NamedSample,
    TernarySample,
    TernarySampleIW,
)
from bolts.data.datasets import ISIC, ColoredMNIST
from bolts.data.datasets.audio.base import PBAudioDataset


@pytest.mark.slow
@pytest.mark.parametrize("ds_cls", [ColoredMNIST, ISIC])
def test_datasets(ds_cls: type[VisionDataset]) -> None:
    """Basic test for datasets.
    Confirms that the datasets can be instantiated and have a functional __getitem__ method.
    """
    transform = T.ToTensor()
    ds = ds_cls(root=Path("~/Data").expanduser(), transform=transform)
    for _ds in ds:
        assert _ds[0] is not None


def test_audio_dataset() -> None:
    """Tests basic functionality of an audio dataset base class."""
    x = torch.rand(1, 10)
    audio_dir = Path(r"Sample path")
    dataset = PBAudioDataset(x=x, audio_dir=audio_dir)

    assert dataset is not None
    assert len(dataset) == len(x)



def test_add_field() -> None:
    x = torch.rand(3, 2)
    s = torch.randint(0, 2, (1, 2))
    y = torch.randint(0, 2, (1, 2))
    iw = torch.rand(1, 2)
    ns = NamedSample(x)
    assert isinstance(ns.add_field(), NamedSample)
    assert isinstance(ns.add_field(y=y), BinarySample)
    assert isinstance(ns.add_field(y=y, iw=iw), BinarySampleIW)
    assert isinstance(ns.add_field(y=y, s=s), TernarySample)
    assert isinstance(ns.add_field(y=y, s=s, iw=iw), TernarySampleIW)
    bs = BinarySample(x=x, y=y)
    assert isinstance(bs.add_field(), BinarySample)
    assert isinstance(bs.add_field(iw=iw), BinarySampleIW)
    assert isinstance(bs.add_field(s=s), TernarySample)
    assert isinstance(bs.add_field(s=s, iw=iw), TernarySampleIW)
    bsi = BinarySampleIW(x=x, y=y, iw=iw)
    assert isinstance(bsi.add_field(), BinarySampleIW)
    assert isinstance(bsi.add_field(s=s), TernarySampleIW)
    ts = TernarySample(x=x, s=s, y=y)
    assert isinstance(ts.add_field(), TernarySample)
    assert isinstance(ts.add_field(iw=iw), TernarySampleIW)
    tsi = TernarySampleIW(x=x, s=s, y=y, iw=iw)
    assert isinstance(tsi.add_field(), TernarySampleIW)
