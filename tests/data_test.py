from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
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
from bolts.data.datasets import ISIC, ColoredMNIST, Ecoacoustics


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


@pytest.mark.slow
def test_audio_dataset() -> None:
    root_dir = Path("~/Data").expanduser()
    target_attribute = "habitat"

    ds_cls_dnwld = Ecoacoustics(root=root_dir, target_attr=target_attribute)
    assert ds_cls_dnwld is not None

    ds_cls_no_dnwld = Ecoacoustics(root=root_dir, download=False, target_attr=target_attribute)
    assert ds_cls_no_dnwld is not None

    metadata = pd.read_csv(root_dir / "Ecoacoustics" / "metadata.csv")

    # Test __str__
    assert str(ds_cls_no_dnwld).splitlines()[0] == "Dataset Ecoacoustics"

    # Test __len__
    num_audio_samples = []
    num_audio_samples.extend(root_dir.glob("**/*.wav"))
    assert len(ds_cls_dnwld) == len(num_audio_samples)
    assert len(ds_cls_no_dnwld) == len(num_audio_samples)

    # Test metadata aligns with labels file.
    audio_samples_to_check = [
        "FS-08_0_20150802_0625.wav",
        "PL-10_0_20150604_0445.wav",
        "KNEPP-02_0_20150510_0730.wav",
    ]
    habitat_target_attributes = ["EC2", "UK1", np.nan]
    for sample, label in zip(audio_samples_to_check, habitat_target_attributes):
        matched_row = metadata.loc[metadata['fileName'] == sample]
        if type(label) == str:
            assert matched_row.iloc[0][target_attribute] == label
        else:
            assert np.isnan(matched_row.iloc[0][target_attribute])

    # Test processed folder
    processed_audio_dir = root_dir / "Ecoacoustics" / "processed_audio"
    assert processed_audio_dir.exists()


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
