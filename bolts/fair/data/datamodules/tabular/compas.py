"""COMPAS Dataset."""

from typing import ClassVar

import ethicml as em

from .base import TabularDataModule

__all__ = ["CompasDataModule"]


class CompasDataModule(TabularDataModule):
    """COMPAS Dataset."""

    NUM_CLASSES: ClassVar[int] = 2
    NUM_SENS: ClassVar[int] = 2

    @property
    def em_dataset(self) -> em.Dataset:
        return em.compas(split="Sex")
