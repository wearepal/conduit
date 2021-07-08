"""Common to all datamodules."""
import logging
from typing import Optional, Sequence, Union

from kit import implements
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Sampler

__all__ = ["BaseDataModule"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule for both Tabular and Vision data-modules."""

    _train_data: Dataset
    _val_data: Dataset
    _test_data: Dataset

    def __init__(
        self,
        batch_size: int,
        val_prop: float,
        test_prop: float,
        num_workers: int,
        seed: int,
        persist_workers: bool,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory

    @property
    def train_prop(self) -> float:
        return 1 - (self.val_prop + self.test_prop)

    def make_dataloader(
        self,
        ds: Dataset,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
    ) -> DataLoader:
        """Make DataLoader."""
        return DataLoader(
            ds,
            batch_size=1 if batch_sampler is not None else self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
        )

    @implements(pl.LightningDataModule)
    def train_dataloader(self, shuffle: bool = False, drop_last: bool = True) -> DataLoader:
        return self.make_dataloader(self._train_data, shuffle=shuffle, drop_last=drop_last)

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self._val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(self._test_data)
