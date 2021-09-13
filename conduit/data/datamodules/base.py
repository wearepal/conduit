"""Base class from which all data-modules in conduit inherit."""
from __future__ import annotations
from abc import abstractmethod
import logging
from typing import Optional, Sequence, Union

from kit import implements
from kit.misc import str_to_enum
from kit.torch import SequentialBatchSampler, StratifiedBatchSampler, TrainingMode
from kit.torch.data import num_batches_per_epoch
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset
from typing_extensions import final

from conduit.data.datasets.base import CdtDataset
from conduit.data.datasets.utils import (
    CdtDataLoader,
    extract_base_dataset,
    get_group_ids,
)
from conduit.data.datasets.wrappers import InstanceWeightedDataset
from conduit.data.structures import ImageSize, TrainValTestSplit
from conduit.types import Stage

__all__ = ["CdtDataModule"]


class CdtDataModule(pl.LightningDataModule):
    """Base DataModule for both Tabular and Vision data-modules."""

    _logger: logging.Logger | None = None

    def __init__(
        self,
        *,
        train_batch_size: int = 64,
        eval_batch_size: Optional[int] = 100,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        num_workers: int = 0,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = "epoch",
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size if eval_batch_size is None else eval_batch_size
        self.num_workers = num_workers
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory
        self.stratified_sampling = stratified_sampling
        self.instance_weighting = instance_weighting
        if isinstance(training_mode, str):
            training_mode = str_to_enum(str_=training_mode, enum=TrainingMode)
        self.training_mode = training_mode

        self._train_data_base: Dataset | None = None
        self._val_data_base: Dataset | None = None
        self._test_data_base: Dataset | None = None

        self._train_data: Dataset | None = None
        self._val_data: Dataset | None = None
        self._test_data: Dataset | None = None
        # Information (cardinality/dimensionality) about the data
        self._card_s: int | None = None
        self._card_y: int | None = None
        self._dim_s: torch.Size | None = None
        self._dim_y: torch.Size | None = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    @property
    def train_prop(self) -> float:
        return 1 - (self.val_prop + self.test_prop)

    def make_dataloader(
        self,
        ds: Dataset,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Sampler[Sequence[int]] | None = None,
    ) -> DataLoader:
        """Make DataLoader."""
        return CdtDataLoader(
            ds,
            batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
        )

    @property
    @final
    def train_data_base(self) -> Dataset:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.train_data_base' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        return self._train_data_base

    @property
    @final
    def train_data(self) -> Dataset:
        if self._train_data is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.train_data' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        return self._train_data

    @property
    @final
    def val_data(self) -> Dataset:
        if self._val_data is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.val_data' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        return self._val_data

    @property
    @final
    def test_data(self) -> Dataset:
        if self._test_data is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.test_data' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        return self._test_data

    def train_dataloader(
        self, *, shuffle: bool = False, drop_last: bool = False, batch_size: int | None = None
    ) -> DataLoader:
        batch_size = self.train_batch_size if batch_size is None else batch_size

        if self.stratified_sampling:
            group_ids = get_group_ids(self.train_data)
            num_groups = len(group_ids.unique())
            num_samples_per_group = batch_size // num_groups
            if batch_size % num_groups:
                self.log(
                    f"For stratified sampling, the batch size must be a multiple of the number of groups."
                    f"Since the batch size is not integer divisible by the number of groups ({num_groups}),"
                    f"the batch size is being reduced to {num_samples_per_group * num_groups}."
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
        return self.make_dataloader(
            batch_size=self.train_batch_size, ds=self.train_data, batch_sampler=batch_sampler
        )

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(batch_size=self.eval_batch_size, ds=self.val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(batch_size=self.eval_batch_size, ds=self.test_data)

    @property
    @final
    @implements(pl.LightningDataModule)
    def dims(self) -> tuple[int, ...]:
        if self._dims:
            return self._dims
        if self._train_data is not None:
            input_size = self._train_data[0].x.shape  # type: ignore
            if len(input_size) == 3:
                input_size = ImageSize(*input_size)
            else:
                input_size = tuple(input_size)
            self._dims = input_size
            return self._dims
        cls_name = self.__class__.__name__
        raise AttributeError(
            f"'{cls_name}.size' cannot be determined because 'setup' has not yet been called."
        )

    @final
    def _num_samples(self, dataset: Dataset) -> int:
        if hasattr(dataset, "__len__"):
            return len(dataset)  # type: ignore
        raise AttributeError(
            f"Number of samples cannot be determined as dataset of type '{dataset.__class__.__name__}' "
            "has no '__len__' attribute defined."
        )

    @property
    @final
    def num_train_samples(self) -> int:
        return self._num_samples(self.train_data)

    @property
    @final
    def num_val_samples(self) -> int:
        return self._num_samples(self.val_data)

    @property
    @final
    def num_test_samples(self) -> int:
        return self._num_samples(self.test_data)

    @final
    def num_train_batches(self, drop_last: bool = False) -> int:
        if self.training_mode is TrainingMode.step:
            raise AttributeError(
                "'num_train_batches' can only be computed when 'training_mode' is set to 'epoch'."
            )
        return num_batches_per_epoch(
            num_samples=self.num_train_samples,
            batch_size=self.train_batch_size,
            drop_last=drop_last,
        )

    @property
    @final
    def dim_y(self) -> tuple[int, ...]:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.dim_y' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        if not isinstance(self._train_data_base, CdtDataset):
            raise AttributeError(
                f"'dim_y' can only determined for {CdtDataset.__name__} instances."
            )
        return self._train_data_base.dim_y

    @property
    @final
    def dim_s(self) -> tuple[int, ...]:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.dim_s' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        if not isinstance(self._train_data_base, CdtDataset):
            raise AttributeError(
                f"'dim_s' can only determined for {CdtDataset.__name__} instances."
            )
        return self._train_data_base.dim_s

    @property
    @final
    def card_y(self) -> int:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.card_y' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        if not isinstance(self._train_data_base, CdtDataset):
            raise AttributeError(
                f"'card_y' can only determined for {CdtDataset.__name__} instances."
            )
        return self._train_data_base.card_y

    @property
    @final
    def card_s(self) -> int:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.card_s' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        if not isinstance(self._train_data_base, CdtDataset):
            raise AttributeError(
                f"'card_s' can only determined for {CdtDataset.__name__} instances."
            )
        return self._train_data_base.card_s

    @abstractmethod
    def _get_splits(self) -> TrainValTestSplit:
        ...

    @implements(pl.LightningDataModule)
    @final
    def setup(self, stage: Stage | None = None, force_reset: bool = False) -> None:
        # Only perform the setup if it hasn't already been done
        if force_reset or (self._train_data is None):
            self._setup(stage=stage)
            self._post_setup()

    def _setup(self, stage: Stage | None = None) -> None:
        train_data, self._val_data, self._test_data = self._get_splits()
        if self.instance_weighting:
            train_data = InstanceWeightedDataset(train_data)
        self._train_data = train_data

    def _post_setup(self) -> None:
        # Make information (cardinality/dimensionality) about the dataset directly accessible through the data-module
        self._train_data_base = extract_base_dataset(
            dataset=self.train_data, return_subset_indices=False
        )
        self._val_data_base = extract_base_dataset(
            dataset=self.val_data, return_subset_indices=False
        )
        self._test_data_base = extract_base_dataset(
            dataset=self.test_data, return_subset_indices=False
        )
