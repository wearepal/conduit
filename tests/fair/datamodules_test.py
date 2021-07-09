"""Test DataModules."""
from __future__ import annotations
from pathlib import Path

import pytest
import torch
from typing_extensions import Final

from bolts.data.datamodules.base import PBDataModule
from bolts.data.datamodules.vision.base import PBVisionDataModule
from bolts.data.datamodules.vision.celeba import CelebADataModule
from bolts.fair.data.datamodules import AdultDataModule, CompasDataModule

BATCHSIZE: Final[int] = 4


def _create_dm(dm_cls: type[PBDataModule], stratified: bool) -> PBDataModule:
    dm_kwargs = dict(
        batch_size=BATCHSIZE,
        stratified_sampling=stratified,
        root=Path("~/Data").expanduser(),
    )
    if isinstance(dm_cls, PBVisionDataModule):
        dm_kwargs["root"] = Path("~/Data").expanduser()
    dm = dm_cls(**dm_kwargs)
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.mark.slow
@pytest.mark.parametrize("stratified", [True, False])
@pytest.mark.parametrize("dm_cls", [AdultDataModule, CompasDataModule])
def test_data_modules(dm_cls: type[PBDataModule], stratified: bool) -> None:
    """Test the datamodules."""
    dm = _create_dm(dm_cls, stratified)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([BATCHSIZE, *dm.size()])  # type: ignore
    assert batch.s.size() == torch.Size([BATCHSIZE, 1])
    assert batch.y.size() == torch.Size([BATCHSIZE, 1])


@pytest.mark.slow
def test_persist_param() -> None:
    """Test that the loader works with persist_workers flag."""
    dm = CelebADataModule(root="data", persist_workers=True, num_workers=1)
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([32, *dm.size()])  # type: ignore
    assert batch.s.size() == torch.Size([32, 1])
    assert batch.y.size() == torch.Size([32, 1])
