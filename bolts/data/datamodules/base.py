"""Base class from which all data-modules in palbolts inherit."""
from __future__ import annotations
import logging
from typing import Sequence

from kit import implements
from kit.torch import SequentialBatchSampler, StratifiedBatchSampler, TrainingMode
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset

from bolts.common import Stage
from bolts.data.datasets.utils import get_group_ids, pb_default_collate
from bolts.data.datasets.wrappers import InstanceWeightedDataset
from bolts.data.structures import TrainValTestSplit

__all__ = ["PBDataModule"]


class PBDataModule(pl.LightningDataModule):
    """Base DataModule for both Tabular and Vision data-modules."""

    _train_data: Dataset
    _val_data: Dataset
    _test_data: Dataset
    _logger: logging.Logger | None = None

    def __init__(
        self,
        *,
        batch_size: int = 64,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        num_workers: int = 0,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: TrainingMode = TrainingMode.epoch,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory
        self.stratified_sampling = stratified_sampling
        self.instance_weighting = instance_weighting
        self.training_mode = training_mode

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log(self, msg: str) -> None:
        self._logger.info(msg)

    @property
    def train_prop(self) -> float:
        return 1 - (self.val_prop + self.test_prop)

    def make_dataloader(
        self,
        ds: Dataset,
        *,
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
            collate_fn=pb_default_collate,
        )

    def train_dataloader(
        self, *, shuffle: bool = False, drop_last: bool = False, batch_size: int | None = None
    ) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.stratified_sampling:
            group_ids = get_group_ids(self._train_data)
            num_groups = len(group_ids.unique())
            num_samples_per_group = batch_size // num_groups
            if batch_size % num_groups:
                self.log(
                    f"For stratified sampling, the batch size must be a multiple of the number of groups."
                    "Since the batch size is not integer divisible by the number of groups ({num_groups}),"
                    "the batch size is being reduced to {num_samples_per_group * num_groups}."
                )
            batch_sampler = StratifiedBatchSampler(
                group_ids=group_ids.squeeze().tolist(),
                num_samples_per_group=num_samples_per_group,
                shuffle=shuffle,
                base_sampler="sequential",
                training_mode=self.training_mode,
                drop_last=drop_last,
            )
        else:
            batch_sampler = SequentialBatchSampler(
                data_source=self._train_data,  # type: ignore
                batch_size=batch_size,
                shuffle=shuffle,
                training_mode=self.training_mode,
                drop_last=drop_last,
            )
        return self.make_dataloader(ds=self._train_data, batch_sampler=batch_sampler)

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(ds=self._val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(ds=self._test_data)

    def _get_splits(self) -> TrainValTestSplit:
        ...

    @implements(pl.LightningDataModule)
    def setup(self, stage: Stage | None = None) -> None:
        train, self._val_data, self._test_data = self._get_splits()
        if self.instance_weighting:
            train = InstanceWeightedDataset(train)
        self._train_data = train
