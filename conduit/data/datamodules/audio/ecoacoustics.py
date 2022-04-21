"""Ecoacoustics data-module."""
from typing import Any, List

import albumentations as A  # type: ignore
import attr
from pytorch_lightning import LightningDataModule
from ranzen import implements

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics, SoundscapeAttr
from conduit.data.structures import TernarySample, TrainValTestSplit
from conduit.transforms.audio import Framing, LogMelSpectrogram

from .base import CdtAudioDataModule

__all__ = ["EcoacousticsDataModule"]


@attr.define(kw_only=True)
class EcoacousticsDataModule(CdtAudioDataModule[Ecoacoustics, TernarySample]):
    """Data-module for the Ecoacoustics dataset."""

    segment_len: float = 15
    sample_rate: int = 48_000
    target_attrs: List[SoundscapeAttr]

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Ecoacoustics(
            root=self.root,
            download=True,
            segment_len=self.segment_len,
            target_attrs=self.target_attrs,
        )

    @property
    def _default_transform(self) -> A.Compose:
        return A.Compose(
            [
                LogMelSpectrogram(),
                Framing(),
            ]
        )

    @property  # type: ignore[misc]
    @implements(CdtAudioDataModule)
    def _default_train_transforms(self) -> A.Compose:
        return self._default_transform

    @property  # type: ignore[misc]
    @implements(CdtAudioDataModule)
    def _default_test_transforms(self) -> A.Compose:
        return self._default_transform

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit[Ecoacoustics]:
        all_data = Ecoacoustics(
            root=self.root,
            transform=None,  # Transform is applied in `CdtAudioDataModule._setup`
            segment_len=self.segment_len,
            target_attrs=self.target_attrs,
            sample_rate=self.sample_rate,
            download=False,
        )

        val_data, test_data, train_data = all_data.random_split(
            props=(self.val_prop, self.test_prop), seed=self.seed
        )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
