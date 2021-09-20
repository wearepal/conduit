"""Base class for audio datasets."""
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import albumentations as A
import attr
from kit import implements

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.utils import AlbumentationsTform
from conduit.types import Stage

__all__ = ["CdtAudioDataModule"]


@attr.define(kw_only=True)
class CdtAudioDataModule(CdtDataModule):

    root: Union[str, Path] = attr.field(kw_only=False)

    @property
    @abstractmethod
    def _base_augmentations(self) -> A.Compose:
        ...

    @property
    @abstractmethod
    def _train_augmentations(self) -> A.Compose:
        ...

    @implements(CdtDataModule)
    def _setup(self, stage: Optional[Stage] = None) -> None:
        train, val, test = self._get_splits()
        self._train_data = train
        self._val_data = val
        self._test_data = test

    def _augmentations(self, train: bool) -> A.Compose:
        # Base augmentations (augmentations that are applied to all splits of the data)
        augs: list[AlbumentationsTform] = [self._base_augmentations]
        # Add training augmentations on top of base augmentations
        if train:
            augs.append(self._train_augmentations)
        return A.Compose(augs)
