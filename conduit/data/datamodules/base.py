"""Base class from which all data-modules in conduit inherit."""
from abc import abstractmethod
import logging
from typing import Optional, Sequence, Tuple

import attr
from kit import implements
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


@attr.s(kw_only=True, auto_attribs=True)
class CdtDataModule(pl.LightningDataModule):
    """Base DataModule for both Tabular and Vision data-modules.

    :param val_prop: Proprtion (float)  of samples to use for the validation split
    :param test_prop: Proportion (float) of samples to use for the test split
    :param num_workers: How many workers to use for loading data
    :param batch_size: How many samples per batch to load
    :param seed: RNG Seed
    :param persist_workers: Use persistent workers in dataloader?
    :param pin_memory: Should the memory be pinned?
    :param stratified_sampling: Use startified sampling?
    :param stratified_sampling: Use instance-weighting?
    :param training mode: Which training mode to use ('epoch' vs. 'step').
    """

    train_batch_size: int = 64
    eval_batch_size: int = train_batch_size
    val_prop: float = 0.2
    test_prop: float = 0.2
    num_workers: int = 0
    seed: int = 47
    persist_workers: bool = False
    pin_memory: bool = True
    stratified_sampling: bool = False
    instance_weighting: bool = False

    training_mode: TrainingMode = TrainingMode.epoch

    _logger: Optional[logging.Logger] = attr.field(default=None, init=False)

    _train_data_base: Optional[Dataset] = attr.field(default=None, init=False)
    _val_data_base: Optional[Dataset] = attr.field(default=None, init=False)
    _test_data_base: Optional[Dataset] = attr.field(default=None, init=False)

    _train_data: Optional[Dataset] = attr.field(default=None, init=False)
    _val_data: Optional[Dataset] = attr.field(default=None, init=False)
    _test_data: Optional[Dataset] = attr.field(default=None, init=False)
    _card_s: Optional[int] = attr.field(default=None, init=False)
    _card_y: Optional[int] = attr.field(default=None, init=False)
    _dim_s: Optional[torch.Size] = attr.field(default=None, init=False)
    _dim_y: Optional[torch.Size] = attr.field(default=None, init=False)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    @property
    def train_prop(self) -> float:
        return- (self.val_prop + self.test_prop)

    def make_dataloader(
        self,
        ds: Dataset,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
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
                "not yet been called."
            )
        return self._train_data_base

    @property
    @final
    def train_data(self) -> Dataset:
        if self._train_data is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.train_data' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called."
            )
        return self._train_data

    @property
    @final
    def val_data(self) -> Dataset:
        if self._val_data is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.val_data' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called."
            )
        return self._val_data

    @property
    @final
    def test_data(self) -> Dataset:
        if self._test_data is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.test_data' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called."
            )
        return self._test_data

    def train_dataloader(
        self, *, shuffle: bool = False, drop_last: bool = False, batch_size: Optional[int] = None
    ) -> DataLoader:
        batch_size = self.train_batch_size if batch_size is None else batch_size

        if self.stratified_sampling:
            group_ids = get_group_ids(self.train_data)
            num_groups = len(group_ids.unique())
            num_samples_per_group = batch_size // num_groups
            if batch_size % num_groups:
                self.log(
                    f"For stratified sampling, the batch size must bemultiple of the number of groups."
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
    def dims(self) -> Tuple[int, ...]:
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
    def dim_y(self) -> Tuple[int, ...]:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.dim_y' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called."
            )
        if not isinstance(self._train_data_base, CdtDataset):
            raise AttributeError(
                f"'dim_y' can only determined for {CdtDataset.__name__} instances."
            )
        return self._train_data_base.dim_y

    @property
    @final
    def dim_s(self) -> Tuple[int, ...]:
        if self._train_data_base is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.dim_s' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called."
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
                "not yet been called."
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
                "not yet been called."
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
    def setup(self, stage: Optional[Stage] = None, force_reset: bool = False) -> None:
        # Only perform the setup if it hasn't already been done
        if force_reset or (self._train_data is None):
            self._setup(stage=stage)
            self._post_setup()

    def _setup(self, stage: Optional[Stage] = None) -> None:
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
