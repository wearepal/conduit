"""Tests for models."""
import pytest
import pytorch_lightning as pl
from torch import nn

from fair_bolts.datamodules import CelebaDataModule
from fair_bolts.models.erm_baseline import ErmBaseline
from fair_bolts.models.laftr_baseline import Laftr


@pytest.mark.parametrize("dm_class", [CelebaDataModule])
def test_laftr(
    enc: nn.Module, adv: nn.Module, clf: nn.Module, dec: nn.Module, dm_class: pl.LightningDataModule
) -> None:
    """Test the Laftr model."""
    dm = dm_class()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(fast_dev_run=True)

    model = Laftr(
        enc=enc,
        dec=dec,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        lr_gamma=0.999,
        disc_steps=1,
        fairness="DP",
        recon_weight=1.0,
        clf_weight=0.0,
        adv_weight=1.0,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dm)


@pytest.mark.parametrize("dm_class", [CelebaDataModule])
def test_erm(enc: nn.Module, clf: nn.Module, dm_class: pl.LightningDataModule) -> None:
    """Test the ERM model."""
    dm = dm_class()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(fast_dev_run=True)

    model = ErmBaseline(
        enc=enc,
        clf=clf,
        weight_decay=1e-8,
        lr_gamma=0.999,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dm)


@pytest.mark.parametrize("dm_class", [CelebaDataModule])
def test_kc(enc: nn.Module, clf: nn.Module, dm_class: pl.LightningDataModule) -> None:
    """Test the ERM model."""
    dm = dm_class()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(fast_dev_run=True)

    model = ErmBaseline(
        enc=enc,
        clf=clf,
        weight_decay=1e-8,
        lr_gamma=0.999,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dm)
