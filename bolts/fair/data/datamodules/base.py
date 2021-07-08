"""Common to all datamodules."""
from __future__ import annotations
import logging
from typing import Sequence

from kit import implements
from kit.torch import StratifiedSampler
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Sampler

from bolts.data.datamodules import BaseDataModule as _BaseDataModule

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())

__all__ = ["BaseDataModule"]


class BaseDataModule(_BaseDataModule):
    """Base DataModule of both Tabular and Vision DataModules."""

    def __init__(
        self,
        batch_size: int,
        val_split: float | int,
        test_split: float | int,
        num_workers: int,
        seed: int,
        persist_workers: bool,
        pin_memory: bool,
        stratified_sampling: bool,
        sample_with_replacement: bool,
    ):
        super().__init__(
            batch_size=batch_size,
            val_prop=val_split,
            test_prop=test_split,
            num_workers=num_workers,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
        )
        self.stratified_sampling = stratified_sampling
        self.sample_with_replacement = sample_with_replacement

    def make_dataloader(
        self,
        ds: Dataset,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Sampler[Sequence[int]] | None = None,
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
        if not self.stratified_sampling:
            return self.make_dataloader(
                self.train_data, shuffle=True, drop_last=drop_last, batch_sampler=None
            )
        from bolts.fair.data import extract_labels_from_dataset

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
