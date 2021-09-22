"""Base class for audio datasets."""
from abc import abstractmethod
from pathlib import Path
from typing import Union

import albumentations as A
import attr


from conduit.data.datamodules.base import CdtDataModule


__all__ = ["CdtAudioDataModule"]


@attr.define(kw_only=True)
class CdtAudioDataModule(CdtDataModule):

    root: Union[str, Path] = attr.field(kw_only=False)

    @property
    @abstractmethod
    def _base_augmentations(self) -> A.Compose:
        ...
