"""Ecoacoustics data-module."""
from typing import Any, List, Optional, Sequence

import albumentations as A  # type: ignore
import attr
from pytorch_lightning import LightningDataModule
from ranzen import implements
from torch.utils.data import Sampler

from conduit.data.datamodules.base import CdtDataModule, I
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics, SoundscapeAttr
from conduit.data.structures import BinarySample, D, TernarySample, TrainValTestSplit
from conduit.transforms.audio import Framing, LogMelSpectrogramNp

from .base import CdtAudioDataModule

__all__ = ["EcoacousticsDataModule"]
from conduit.data.datasets.utils import CdtDataLoader


@attr.define(kw_only=True)
class EcoacousticsDataModule(CdtAudioDataModule[Ecoacoustics, TernarySample]):
    """Data-module for the Ecoacoustics dataset."""

    segment_len: float = 15
    sample_rate: int = 48_000
    target_attrs: List[SoundscapeAttr]

    @staticmethod
    def converter(batch: BinarySample) -> BinarySample:
        return BinarySample(x=batch.x, y=batch.y.expand(batch.x.shape[0]))

    def make_dataloader(
        self,
        ds: D,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
    ) -> CdtDataLoader[I]:
        """Make DataLoader."""
        return CdtDataLoader(
            ds,
            batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
            converter=self.converter,
        )

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
                LogMelSpectrogramNp(),
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
