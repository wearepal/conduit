"""Tests for models."""
import pytorch_lightning as pl
from torch import nn

from bolts.models.dann import Dann
from bolts.models.erm import ErmBaseline
from bolts.models.laftr import Laftr


def test_laftr(
    dummy_dm: pl.LightningDataModule,
    enc: nn.Module,
    adv: nn.Module,
    clf: nn.Module,
    dec: nn.Module,
) -> None:
    """Test the Laftr model."""
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

    trainer.fit(model, datamodule=dummy_dm)


def test_dann(
    dummy_dm: pl.LightningDataModule, enc: nn.Module, adv: nn.Module, clf: nn.Module
) -> None:
    """Test the Laftr model."""
    trainer = pl.Trainer(fast_dev_run=True)

    model = Dann(
        enc=enc,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dummy_dm)


def test_erm(dummy_dm: pl.LightningDataModule, enc: nn.Module, clf: nn.Module) -> None:
    """Test the ERM model."""
    trainer = pl.Trainer(fast_dev_run=True)

    model = ErmBaseline(
        enc=enc,
        clf=clf,
        weight_decay=1e-8,
        lr_gamma=0.999,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dummy_dm)


def test_kc(dummy_dm: pl.LightningDataModule, enc: nn.Module, clf: nn.Module) -> None:
    """Test the ERM model."""
    trainer = pl.Trainer(fast_dev_run=True)

    model = ErmBaseline(
        enc=enc,
        clf=clf,
        weight_decay=1e-8,
        lr_gamma=0.999,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dummy_dm)
