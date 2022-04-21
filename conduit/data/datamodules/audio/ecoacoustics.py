"""Ecoacoustics data-module."""
from typing import Any, List

import attr
from pytorch_lightning import LightningDataModule
from ranzen import implements

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics, SoundscapeAttr
from conduit.data.structures import TernarySample, TrainValTestSplit

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
