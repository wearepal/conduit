"""Tests for models."""
from abc import abstractmethod
from typing import List, Optional, Tuple

from ranzen import implements
import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.datasets.wrappers import InstanceWeightedDataset
from conduit.fair.data.datasets import DummyDataset
from conduit.fair.misc import FairnessType
from conduit.fair.models import DANN, GPD, KC, LAFTR, ERMClassifierF, FairMixup


class Mp64x64Net(nn.Module):
    """Predefined 64x64 net."""

    def __init__(self, batch_norm: bool, in_chans: int, target_dim: int) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        self.net = self._build(in_chans=in_chans, target_dim=target_dim)

    def _conv_block(
        self, in_chans: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ) -> List[nn.Module]:
        _block: List[nn.Module] = []
        _block += [
            nn.Conv2d(in_chans, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    def _build(self, in_chans: int, target_dim: int) -> nn.Sequential:
        layers = nn.ModuleList()
        layers.extend(self._conv_block(in_chans, 64, 5, 1, 0))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(64, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 256, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(256, 512, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Flatten()]
        layers += [nn.Linear(512, target_dim)]

        return nn.Sequential(*layers)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class View(nn.Module):
    """Reshape Tensor."""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)


def down_conv(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
) -> nn.Module:
    """Down convolutions."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.SiLU(),
    )


def up_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
) -> nn.Module:
    """Up convolutions."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """Encoder net."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        initial_hidden_channels: int,
        levels: int,
        encoding_dim: int,
    ):
        super().__init__()
        layers = nn.ModuleList()
        c_in, height, width = input_shape
        c_out = initial_hidden_channels

        for level in range(levels):
            if level != 0:
                c_in = c_out
                c_out *= 2
            layers.append(
                nn.Sequential(
                    down_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                    down_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
                )
            )
            height //= 2
            width //= 2

        flattened_size = c_out * height * width
        layers += [nn.Flatten()]
        layers += [nn.Linear(flattened_size, encoding_dim)]

        self.encoder = nn.Sequential(*layers)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder net."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        initial_hidden_channels: int,
        levels: int,
        encoding_dim: int,
        decoding_dim: int,
        decoder_out_act: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        c_in, height, width = input_shape
        c_out = initial_hidden_channels

        for level in range(levels):
            if level != 0:
                c_in = c_out
                c_out *= 2

            layers.append(
                nn.Sequential(
                    # inverted order
                    up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0),
                    down_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
                )
            )

            height //= 2
            width //= 2

        flattened_size = c_out * height * width

        layers += [View((c_out, height, width))]
        layers += [nn.Linear(encoding_dim, flattened_size)]
        layers = layers[::-1]
        layers += [nn.Conv2d(input_shape[0], decoding_dim, kernel_size=1, stride=1, padding=0)]

        if decoder_out_act is not None:
            layers += [decoder_out_act]

        self.decoder = nn.Sequential(*layers)

    @implements(nn.Module)
    def forward(self, z: Tensor, s: Tensor) -> Tensor:
        s = s.view(-1, 1)
        zs = torch.cat([z, s], dim=1)
        return self.decoder(zs)


class EmbeddingClf(nn.Module):
    """Classifier."""

    def __init__(self, encoding_dim: int, out_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(encoding_dim, out_dim))

    @implements(nn.Module)
    def forward(self, z: Tensor) -> Tensor:
        return self.classifier(z)


class DummyBase(pl.LightningDataModule):
    @abstractmethod
    def _get_dl(self) -> DataLoader:
        ...

    def train_dataloader(self) -> DataLoader:
        return self._get_dl()

    def val_dataloader(self) -> DataLoader:
        return self._get_dl()

    def test_dataloader(self) -> DataLoader:
        return self._get_dl()


class DummyDataModule(DummyBase):
    def _get_dl(self) -> DataLoader:
        train_ds = InstanceWeightedDataset(
            DummyDataset((3, 64, 64), (1,), (1,), (1,), num_samples=50)
        )
        return CdtDataLoader(train_ds, batch_size=25, shuffle=False)


class DummyDataModuleDim2(DummyBase):
    def _get_dl(self) -> DataLoader:
        train_ds = InstanceWeightedDataset(
            DummyDataset((3, 64, 64), (1, 1), (1, 1), (1, 1), num_samples=50)
        )
        return CdtDataLoader(train_ds, batch_size=25, shuffle=False)


@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
@pytest.mark.parametrize("fairness", FairnessType)
def test_laftr(dm: pl.LightningDataModule, fairness: FairnessType) -> None:
    """Test the LAFTR model."""
    trainer = pl.Trainer(fast_dev_run=True)

    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    adv = EmbeddingClf(encoding_dim=128, out_dim=1)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    dec = Decoder(
        input_shape=(3, 64, 64),
        initial_hidden_channels=64,
        levels=3,
        encoding_dim=128 + 1,
        decoding_dim=3,
        decoder_out_act=nn.Tanh(),
    )
    model = LAFTR(
        enc=enc,
        dec=dec,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        disc_steps=1,
        fairness=fairness,
        recon_weight=1.0,
        clf_weight=0.0,
        adv_weight=1.0,
        lr=1e-3,
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.gpu
@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
@pytest.mark.parametrize("fairness", FairnessType)
def test_laftr_gpu(dm: pl.LightningDataModule, fairness: FairnessType) -> None:
    """Test the LAFTR model."""
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)

    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    adv = EmbeddingClf(encoding_dim=128, out_dim=1)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    dec = Decoder(
        input_shape=(3, 64, 64),
        initial_hidden_channels=64,
        levels=3,
        encoding_dim=128 + 1,
        decoding_dim=3,
        decoder_out_act=nn.Tanh(),
    )
    model = LAFTR(
        enc=enc,
        dec=dec,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        disc_steps=1,
        fairness=fairness,
        recon_weight=1.0,
        clf_weight=0.0,
        adv_weight=1.0,
        lr=1e-3,
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_dann(dm: pl.LightningDataModule) -> None:
    """Test the LAFTR model."""
    trainer = pl.Trainer(fast_dev_run=True)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    adv = EmbeddingClf(encoding_dim=128, out_dim=2)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = DANN(
        enc=enc,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.gpu
@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_dann_gpu(dm: pl.LightningDataModule) -> None:
    """Test the DANN model."""
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    adv = EmbeddingClf(encoding_dim=128, out_dim=2)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = DANN(
        enc=enc,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.parametrize("fairness", FairnessType)
@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_fairmixup(dm: pl.LightningDataModule, fairness: FairnessType) -> None:
    """Test the LAFTR model."""
    trainer = pl.Trainer(fast_dev_run=True)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = FairMixup(encoder=enc, clf=clf, weight_decay=1e-8, lr=1e-3, fairness=fairness)
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_erm(dm: pl.LightningDataModule) -> None:
    """Test the ERM model."""
    trainer = pl.Trainer(fast_dev_run=True)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = ERMClassifierF(
        encoder=enc,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.gpu
@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_erm_gpu(dm: pl.LightningDataModule) -> None:
    """Test the ERM model."""
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = ERMClassifierF(
        encoder=enc,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_kc(dm: pl.LightningDataModule) -> None:
    """Test the K&C model."""
    trainer = pl.Trainer(fast_dev_run=True)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = KC(
        encoder=enc,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.gpu
@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_kc_gpu(dm: pl.LightningDataModule) -> None:
    """Test the K&C model."""
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = KC(
        encoder=enc,
        clf=clf,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_gpd(dm: pl.LightningDataModule) -> None:
    """Test the K&C model."""
    trainer = pl.Trainer(fast_dev_run=True)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    adv = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = GPD(
        enc=enc,
        clf=clf,
        adv=adv,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


@pytest.mark.gpu
@pytest.mark.parametrize("dm", [DummyDataModule(), DummyDataModuleDim2()])
def test_gpd_gpu(dm: pl.LightningDataModule) -> None:
    """Test the K&C model."""
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = EmbeddingClf(encoding_dim=128, out_dim=2)
    adv = EmbeddingClf(encoding_dim=128, out_dim=2)
    model = GPD(
        enc=enc,
        clf=clf,
        adv=adv,
        weight_decay=1e-8,
        lr=1e-3,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
