"""Test DataModules."""
import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from typing_extensions import Final, Type

from fair_bolts.datamodules.adult_datamodule import AdultDataModule
from fair_bolts.datamodules.celeba_datamodule import CelebaDataModule
from fair_bolts.datamodules.cmnist_datamodule import CmnistDataModule
from fair_bolts.datamodules.compas_datamodule import CompasDataModule

BATCHSIZE: Final[int] = 4


def _create_dm(dm_cls: Type[LightningDataModule], stratified: bool) -> LightningDataModule:
    dm = dm_cls(batch_size=BATCHSIZE, stratified_sampling=stratified)
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.mark.parametrize("stratified", [True, False])
@pytest.mark.parametrize(
    "dm_cls", [AdultDataModule, CompasDataModule, CelebaDataModule, CmnistDataModule]
)
def test_data_modules(dm_cls: Type[LightningDataModule], stratified: bool) -> None:
    """Test the datamodules."""
    dm = _create_dm(dm_cls, stratified)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([BATCHSIZE, *dm.size()])
    assert batch.s.size() == torch.Size([BATCHSIZE, 1])
    assert batch.y.size() == torch.Size([BATCHSIZE, 1])
    F.cross_entropy(torch.rand((BATCHSIZE, dm.num_sens)), batch.s.squeeze(-1))
    F.cross_entropy(torch.rand((BATCHSIZE, dm.num_classes)), batch.y.squeeze(-1))
    assert dm.num_classes
    assert dm.num_sens


def test_cache_param() -> None:
    """Test that the loader works with cache flag."""
    dm = CelebaDataModule(cache_data=True)
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert loader.dataset.dataset.__getitem__.cache_info()
    assert batch.x.size() == torch.Size([32, *dm.size()])
    assert batch.s.size() == torch.Size([32, 1])
    assert batch.y.size() == torch.Size([32, 1])
    F.cross_entropy(torch.rand((32, dm.num_sens)), batch.s.squeeze(-1))
    F.cross_entropy(torch.rand((32, dm.num_classes)), batch.y.squeeze(-1))
    assert dm.num_classes
    assert dm.num_sens


def test_persist_param() -> None:
    """Test that the loader works with persist_workers flag."""
    dm = CelebaDataModule(persist_workers=True, num_workers=1)
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([32, *dm.size()])
    assert batch.s.size() == torch.Size([32, 1])
    assert batch.y.size() == torch.Size([32, 1])
    F.cross_entropy(torch.rand((32, dm.num_sens)), batch.s.squeeze(-1))
    F.cross_entropy(torch.rand((32, dm.num_classes)), batch.y.squeeze(-1))
    assert dm.num_classes
    assert dm.num_sens
