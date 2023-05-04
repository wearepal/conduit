"""Ecoacoustics data-module."""
from typing import Any, List, Optional, Sequence

import attr
from torch.utils.data import Sampler
from typing_extensions import override

from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics, SoundscapeAttr
from conduit.data.datasets.utils import AudioTform, CdtDataLoader
from conduit.data.structures import BinarySample, TrainValTestSplit
from conduit.transforms.audio import Compose, Framing, LogMelSpectrogram

from .base import CdtAudioDataModule

__all__ = ["EcoacousticsDataModule"]


@attr.define(kw_only=True)
class EcoacousticsDataModule(CdtAudioDataModule[Ecoacoustics, BinarySample]):
    """Data-module for the Ecoacoustics dataset."""

    segment_len: float = 15
    sample_rate: int = 48_000
    target_attrs: List[SoundscapeAttr]

    @staticmethod
    def _batch_converter(batch: BinarySample) -> BinarySample:
        return BinarySample(x=batch.x, y=batch.y)

    @override
    def make_dataloader(
        self,
        ds: Ecoacoustics,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
    ) -> CdtDataLoader[BinarySample]:
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
            converter=self._batch_converter,
        )

    @override
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Ecoacoustics(
            root=self.root,
            download=True,
            segment_len=self.segment_len,
            target_attrs=self.target_attrs,
        )

    @property
    def _default_transform(self) -> Compose:
        return Compose([LogMelSpectrogram(), Framing()])

    @property
    @override
    def _default_train_transforms(self) -> AudioTform:
        return self._default_transform

    @property
    @override
    def _default_test_transforms(self) -> AudioTform:
        return self._default_transform

    @override
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
