"""Base class for audio datasets."""
from pathlib import Path
from typing import Union

import attr

from conduit.data.datamodules.base import CdtDataModule

__all__ = ["CdtAudioDataModule"]


@attr.define(kw_only=True)
class CdtAudioDataModule(CdtDataModule):

    root: Union[str, Path] = attr.field(kw_only=False)
