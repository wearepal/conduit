from typing import Type

import pytest
from pytorch_lightning import LightningDataModule
import torch

from bolts.datamodules.mnist_datamodule import MNISTDataModule


def _create_dm(dm_cls: Type[LightningDataModule]) -> LightningDataModule:
    dm = dm_cls(batch_size=2)
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.mark.parametrize("dm_cls", [MNISTDataModule])
def test_data_modules(dm_cls: Type[LightningDataModule]) -> None:
    """Test the datamodules."""
    dm = _create_dm(dm_cls)
    loader = dm.train_dataloader()
    img, _ = next(iter(loader))
    assert img.size() == torch.Size([2, *dm.size()])
    assert dm.num_classes
