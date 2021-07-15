"""Base class from which all data-modules in palbolts inherit."""
from __future__ import annotations
from enum import Enum, auto
import logging
from typing import Any, Sequence

from kit import implements
from kit.torch.data import InfSequentialBatchSampler, StratifiedSampler
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from bolts.data.datasets.utils import (
    SizedStratifiedSampler,
    get_group_ids,
    pb_default_collate,
)
from bolts.data.datasets.wrappers import InstanceWeightedDataset
from bolts.data.structures import TrainValTestSplit

__all__ = ["PBDataModule", "TrainingMode"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class TrainingMode(Enum):
    epoch = auto()
    step = auto()


class PBDataModule(pl.LightningDataModule):
    """Base DataModule for both Tabular and Vision data-modules."""

    _train_data: Dataset
    _val_data: Dataset
    _test_data: Dataset

    def __init__(
        self,
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
    def train_prop(self) -> float:
        return 1 - (self.val_prop + self.test_prop)

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
            collate_fn=pb_default_collate,
        )

    def train_dataloader(
        self, shuffle: bool = False, drop_last: bool = True, batch_size: int | None = None
    ) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.stratified_sampling:
            group_ids = get_group_ids(self._train_data)
            num_groups = len(group_ids.unique())
            num_samples_per_group = batch_size // num_groups
            if batch_size % num_groups:
                LOGGER.info(
                    f"For stratified sampling, the batch size must be a multiple of the number of groups."
                    "Since the batch size is not integer divisible by the number of groups ({num_groups}),"
                    "the batch size is being reduced to {num_samples_per_group * num_groups}."
                )
            sampler_kwargs: dict[str, Any] = dict(
                group_ids.squeeze().tolist(),
                num_samples_per_group=num_samples_per_group,
                shuffle=shuffle,
            )
            if self.training_mode is TrainingMode.epoch:
                sampler_cls = SizedStratifiedSampler
            else:
                sampler_cls = StratifiedSampler
                sampler_kwargs["base_sampler"] = "sequential"
            batch_sampler = sampler_cls(**sampler_kwargs)
        else:
            if self.training_mode is TrainingMode.epoch:
                batch_sampler = BatchSampler(
                    sampler=SequentialSampler(data_source=self._train_data),  # type: ignore
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            else:
                batch_sampler = InfSequentialBatchSampler(
                    data_source=self._train_data, batch_size=batch_size, shuffle=shuffle  # type: ignore
                )
        return self.make_dataloader(self._train_data, batch_sampler=batch_sampler)

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self._val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(self._test_data)

    def _get_splits(self) -> TrainValTestSplit:
        ...

    @implements(pl.LightningDataModule)
    def setup(self, stage: Stage | None = None) -> None:
        train, self._val_data, self._test_data = self._get_splits()
        if self.instance_weighting:
            train = InstanceWeightedDataset(train)
        self._train_data = train
