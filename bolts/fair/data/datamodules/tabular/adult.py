"""Adult Income Dataset."""
from typing import ClassVar

import ethicml as em

from .base import TabularDataModule

__all__ = ["AdultDataModule"]


class AdultDataModule(TabularDataModule):
    """UCI Adult Income Dataset."""

    NUM_CLASSES: ClassVar[int] = 2
    NUM_SENS: ClassVar[int] = 2

    @property
    def em_dataset(self) -> em.Dataset:
        return em.adult(split="Sex", binarize_nationality=True)
