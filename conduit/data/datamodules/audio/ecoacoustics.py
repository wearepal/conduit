"""Ecoacoustics data-module."""

from typing import Any, Optional, Union

import albumentations as A
from kit import implements, parsable
from kit.torch import prop_random_split
from kit.torch.data import TrainingMode
from pytorch_lightning import LightningDataModule

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics
from conduit.data.structures import TrainValTestSplit
from conduit.types import SoundscapeAttr

from .base import CdtAudioDataModule

__all__ = ["EcoacousticsDataModule"]


class EcoacousticsDataModule(CdtAudioDataModule):
    """Data-module for the Ecoacoustics dataset."""

    @parsable
    def __init__(
        self,
        root: str,
        *,
        image_size: int = 224,
        train_batch_size: int = 32,
        eval_batch_size: Optional[int] = 64,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        class_train_props: Optional[dict] = None,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = "epoch",
        specgram_segment_len: float = 15,
        num_freq_bins: int = 120,
        target_attr: Union[SoundscapeAttr, str] = SoundscapeAttr.habitat,
    ) -> None:
        super().__init__(
            root=root,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_prop=val_prop,
            test_prop=test_prop,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            stratified_sampling=stratified_sampling,
            instance_weighting=instance_weighting,
            training_mode=training_mode,
        )
        self.image_size = image_size
        self.class_train_props = class_train_props
        self.specgram_segment_len = specgram_segment_len
        self.num_freq_bins = num_freq_bins
        self.target_attr = target_attr

    @property  # type: ignore[misc]
    @implements(CdtAudioDataModule)
    def _base_augmentations(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
            ]
        )

    @property  # type: ignore[misc]
    @implements(CdtAudioDataModule)
    def _train_augmentations(self) -> A.Compose:
        return A.Compose([])

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Ecoacoustics(
            root=self.root,
            download=True,
            specgram_segment_len=self.specgram_segment_len,
            num_freq_bins=self.num_freq_bins,
            target_attr=self.target_attr,
        )

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        all_data = Ecoacoustics(root=self.root, transform=None)

        val_data, test_data, train_data = prop_random_split(
            dataset=all_data, props=(self.val_prop, self.test_prop)
        )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
