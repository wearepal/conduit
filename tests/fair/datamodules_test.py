"""Test DataModules."""
from __future__ import annotations
from pathlib import Path

import pytest
import torch
from typing_extensions import Final

from bolts.data.datamodules.base import PBDataModule
from bolts.data.datamodules.vision.celeba import CelebADataModule
from bolts.fair.data import CrimeDataModule, HealthDataModule
from bolts.fair.data.datamodules import AdultDataModule, CompasDataModule
from bolts.fair.data.datamodules.tabular.admissions import AdmissionsDataModule
from bolts.fair.data.datamodules.tabular.credit import CreditDataModule

BATCHSIZE: Final[int] = 4


def _create_dm(dm_cls: type[PBDataModule], stratified: bool) -> PBDataModule:
    dm_kwargs = dict(
        batch_size=BATCHSIZE,
        stratified_sampling=stratified,
    )
    dm = dm_cls(**dm_kwargs)  # type: ignore[arg-type]
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.mark.parametrize("stratified", [True, False])
@pytest.mark.parametrize(
    "dm_cls",
    [
        AdmissionsDataModule,
        AdultDataModule,
        CompasDataModule,
        CreditDataModule,
        CrimeDataModule,
        HealthDataModule,
    ],
)
def test_data_modules(dm_cls: type[PBDataModule], stratified: bool) -> None:
    """Test the datamodules."""
    dm = _create_dm(dm_cls, stratified)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([BATCHSIZE, *dm.size()])  # type: ignore
    assert batch.s.size() == torch.Size([BATCHSIZE])
    assert batch.y.size() == torch.Size([BATCHSIZE])


@pytest.mark.slow
def test_persist_param() -> None:
    """Test that the loader works with persist_workers flag."""
    dm = CelebADataModule(root=Path("~/Data").expanduser(), persist_workers=True, num_workers=1)
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([32, *dm.size()])  # type: ignore
    assert batch.s.size() == torch.Size([32])
    assert batch.y.size() == torch.Size([32])
