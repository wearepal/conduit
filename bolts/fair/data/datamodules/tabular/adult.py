"""Adult Income Dataset."""
import ethicml as em

from .base import TabularDataModule

__all__ = ["AdultDataModule"]


class AdultDataModule(TabularDataModule):
    """UCI Adult Income Dataset."""

    @property
    def em_dataset(self) -> em.Dataset:
        return em.adult(split="Sex", binarize_nationality=True)
