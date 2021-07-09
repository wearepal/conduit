"""COMPAS Dataset."""
import ethicml as em

from .base import TabularDataModule

__all__ = ["CompasDataModule"]


class CompasDataModule(TabularDataModule):
    """COMPAS Dataset."""

    @property
    def em_dataset(self) -> em.Dataset:
        return em.compas(split="Sex")
