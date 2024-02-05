"""Test DataModules."""

from functools import partial
from pathlib import Path
from typing import Any, Final, Type

import ethicml as em
import pytest
import torch

from conduit.data.datamodules.vision.celeba import CelebADataModule
from conduit.fair.data import CrimeDataModule, HealthDataModule
from conduit.fair.data.datamodules import AdultDataModule, CompasDataModule
from conduit.fair.data.datamodules.tabular.admissions import AdmissionsDataModule
from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule
from conduit.fair.data.datamodules.tabular.credit import CreditDataModule
from conduit.fair.data.datamodules.tabular.german import GermanDataModule
from conduit.fair.data.datamodules.tabular.law import LawDataModule
from conduit.fair.data.datamodules.tabular.sqf import SqfDataModule

BATCHSIZE: Final[int] = 4


def _create_dm(
    dm_cls: Type[EthicMlDataModule],
    stratified: bool = False,
    extra_args: dict[str, Any] | None = None,
) -> EthicMlDataModule:
    extra_args = {} if extra_args is None else extra_args
    dm_partial = partial(dm_cls, train_batch_size=BATCHSIZE, stratified_sampling=stratified)
    dm = dm_partial(**extra_args)
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
def test_data_modules(dm_cls: Type[EthicMlDataModule], stratified: bool) -> None:
    """Test the datamodules."""
    dm = _create_dm(dm_cls, stratified)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([BATCHSIZE, *dm.size()])
    assert batch.s.size() == torch.Size([BATCHSIZE])
    assert batch.y.size() == torch.Size([BATCHSIZE])

    dm_2 = _create_dm(dm_cls, stratified, extra_args={"invert_s": True})
    loader = dm_2.train_dataloader()
    batch_2 = next(iter(loader))

    torch.testing.assert_close(batch.x, batch_2.x)
    torch.testing.assert_close(batch.y, batch_2.y)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(batch.s, batch_2.s, check_dtype=False, check_stride=False)


@pytest.mark.parametrize(
    "dm_cls",
    [
        AdmissionsDataModule,
        AdultDataModule,
        CompasDataModule,
        CreditDataModule,
        CrimeDataModule,
        GermanDataModule,
        HealthDataModule,
        LawDataModule,
        SqfDataModule,
    ],
)
def test_data_modules_props(dm_cls: Type[EthicMlDataModule]) -> None:
    """Test the datamodules."""
    dm = _create_dm(dm_cls)

    assert dm.dim_s == (1,)
    assert dm.dim_y == (1,)
    assert dm.card_s == 2
    assert dm.card_y == 2
    assert len(dm.size()) == 1

    assert isinstance(dm.train_datatuple, em.DataTuple)
    assert isinstance(dm.val_datatuple, em.DataTuple)
    assert isinstance(dm.test_datatuple, em.DataTuple)

    assert dm.feature_groups is not None
    assert isinstance(dm.disc_features, list)
    if len(dm.disc_features) > 0:
        assert isinstance(dm.disc_features[0], str)


@pytest.mark.slow
def test_persist_param(root: Path) -> None:
    """Test that the loader works with persist_workers flag."""
    dm = CelebADataModule(root=root, persist_workers=True, num_workers=1, train_batch_size=32)
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([32, *dm.size()])
    assert batch.s.size() == torch.Size([32])
    assert batch.y.size() == torch.Size([32])
