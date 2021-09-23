"""Ecoacoustics data-module."""

from typing import Any, Optional, Union

import albumentations as A
import attr
from kit import implements
from kit.torch import prop_random_split
from pytorch_lightning import LightningDataModule
import torchaudio.transforms as T

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics
from conduit.data.datasets.utils import AudioTform
from conduit.data.structures import TrainValTestSplit
from conduit.types import SoundscapeAttr

from .base import CdtAudioDataModule

__all__ = ["EcoacousticsDataModule"]


@attr.define(kw_only=True)
class EcoacousticsDataModule(CdtAudioDataModule):
    """Data-module for the Ecoacoustics dataset."""

    specgram_segment_len: float = 15
    target_attr: Union[SoundscapeAttr, str] = SoundscapeAttr.habitat
    resample_rate: int = 22050
    preprocessing_transform: AudioTform = attr.field(
        factory=lambda: T.MelSpectrogram(sample_rate=22050, n_fft=160)
    )

    @property  # type: ignore[misc]
    @implements(CdtAudioDataModule)
    def _base_augmentations(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
            ]
        )

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Ecoacoustics(
            root=self.root,
            download=True,
            specgram_segment_len=self.specgram_segment_len,
            preprocessing_transform=self.preprocessing_transform,
            target_attr=self.target_attr,
            resample_rate=self.resample_rate,
        )

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        all_data = Ecoacoustics(
            root=self.root, transform=None, preprocessing_transform=self.preprocessing_transform
        )

        val_data, test_data, train_data = prop_random_split(
            dataset=all_data, props=(self.val_prop, self.test_prop)
        )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)