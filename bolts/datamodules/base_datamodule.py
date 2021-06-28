"""Common to all datamodules."""
import logging
from abc import abstractmethod
from typing import List, Optional, Sequence, Union

import pytorch_lightning as pl
from kit import implements
from kit.torch import StratifiedSampler
from torch.utils.data import DataLoader, Dataset, Sampler

from bolts.datamodules.utils import extract_labels_from_dataset

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule of both Tabular and Vision DataModules."""

    def __init__(
        self,
        batch_size: int,
        val_split: Union[float, int],
        test_split: Union[float, int],
        num_workers: int,
        seed: int,
        persist_workers: bool,
        pin_memory: bool,
        stratified_sampling: bool,
        sample_with_replacement: bool,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory
        self.stratified_sampling = stratified_sampling
        self.sample_with_replacement = sample_with_replacement

    @staticmethod
    def _get_splits(train_len: int, val_split: Union[int, float]) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len -= val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * train_len)
            train_len -= val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")

        return splits

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
        if self.stratified_sampling:
            s_all, y_all = extract_labels_from_dataset(self._train_data)
            group_ids = (y_all * len(s_all.unique()) + s_all).squeeze()
            num_groups = len(group_ids.unique())
            num_samples_per_group = self.batch_size // num_groups
            if self.batch_size % num_groups:
                LOGGER.info(
                    f"For stratified sampling, the batch size must be a multiple of the number of groups."
                    "Since the batch size is not integer divisible by the number of groups ({num_groups}),"
                    "the batch size is being reduced to {num_samples_per_group * num_groups}."
                )
            batch_sampler = StratifiedSampler(
                group_ids.squeeze().tolist(),
                num_samples_per_group=num_samples_per_group,
                replacement=self.sample_with_replacement,
            )
            return self.make_dataloader(
                self.train_data, batch_sampler=batch_sampler, shuffle=False, drop_last=False
            )
        else:
            return self.make_dataloader(
                self.train_data, shuffle=True, drop_last=drop_last, batch_sampler=None
            )

    @implements(pl.LightningDataModule)
    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.test_data)

    @property
    @abstractmethod
    def train_data(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def val_data(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def test_data(self) -> Dataset:
        ...
