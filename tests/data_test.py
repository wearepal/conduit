from __future__ import annotations
from pathlib import Path

import pytest
from torchvision import transforms as T
from torchvision.datasets import VisionDataset

from bolts.data.datasets import ISIC, MNIST


@pytest.mark.slow
@pytest.mark.parametrize("ds_cls", [MNIST, ISIC])
def test_datasets(ds_cls: type[VisionDataset]) -> None:
    """Basic test for datasets.
    Confirms that the datasets can be instantiated and have a functional __getitem__ method.
    """
    transform = T.ToTensor()
    ds = ds_cls(root=Path("~/Data").expanduser(), transform=transform)
    for _ds in ds:
        assert _ds[0] is not None
