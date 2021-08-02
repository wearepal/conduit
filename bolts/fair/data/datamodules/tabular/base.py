"""Tabular data-module."""
from __future__ import annotations
from abc import abstractmethod
from typing import Optional, Union

import ethicml as em
from ethicml.preprocessing.scaling import ScalerType
from kit import implements
from kit.torch import TrainingMode
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler

from bolts.data.datamodules import PBDataModule
from bolts.data.structures import TrainValTestSplit
from bolts.fair.data.datasets import DataTupleDataset

__all__ = ["TabularDataModule"]


class TabularDataModule(PBDataModule):
    """Base data-module for tabular datasets."""

    def __init__(
        self,
        *,
        train_batch_size: int = 100,
        eval_batch_size: Optional[int] = 256,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 0,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        scaler: ScalerType | None = None,
        training_mode: Union[TrainingMode, str] = TrainingMode.epoch,
    ) -> None:
        """Base data-module for tabular data.

        Args:
            val_prop: Proprtion (float)  of samples to use for the validation split
            test_prop: Proportion (float) of samples to use for the test split
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            seed: RNG Seed
            scaler: SKLearn style data scaler. Fit to train, applied to val and test.
            persist_workers: Use persistent workers in dataloader?
            pin_memory: Should the memory be pinned?
            stratified_sampling: Use startified sampling?
        """
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            seed=seed,
            test_prop=test_prop,
            val_prop=val_prop,
            stratified_sampling=stratified_sampling,
            instance_weighting=instance_weighting,
            training_mode=training_mode,
        )
        self.scaler = scaler if scaler is not None else StandardScaler()
        self._train_datatuple: em.DataTuple | None = None
        self._val_datatuple: em.DataTuple | None = None
        self._test_datatuple: em.DataTuple | None = None
        self._cont_features: list[str] | None = None
        self._disc_features: list[str] | None = None
        self._feature_groups: dict[str, list[slice]] | None = None

    @property
    @abstractmethod
    def em_dataset(self) -> em.Dataset:
        ...

    @staticmethod
    def _get_split_sizes(train_len: int, *, test_prop: int | float) -> list[int]:
        """Computes split sizes for train and validation sets."""
        if isinstance(test_prop, int):
            train_len -= test_prop
            splits = [train_len, test_prop]
        elif isinstance(test_prop, float):
            test_len = int(test_prop * train_len)
            train_len -= test_len
            splits = [train_len, test_len]
        else:
            raise ValueError(f"Unsupported type {type(test_prop)}")

        return splits

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        self.dims = (
            len(self.em_dataset.discrete_features) + len(self.em_dataset.continuous_features),
        )
        self.make_feature_groups()

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        self.datatuple = self.em_dataset.load(ordered=True)

        data_len = int(self.datatuple.x.shape[0])
        num_train_val, num_test = self._get_split_sizes(data_len, test_prop=self.test_prop)
        train_val, test_data = em.train_test_split(
            data=self.datatuple,
            train_percentage=(1 - (num_test / data_len)),
            random_seed=self.seed,
        )
        _, num_val = self._get_split_sizes(num_train_val, test_prop=self.val_prop)
        train_data, val_data = em.train_test_split(
            data=train_val,
            train_percentage=(1 - (num_val / num_train_val)),
            random_seed=self.seed,
        )

        self._train_datatuple, self.scaler = em.scale_continuous(
            self.em_dataset, datatuple=train_data, scaler=self.scaler  # type: ignore
        )
        self._val_datatuple, _ = em.scale_continuous(
            self.em_dataset, datatuple=val_data, scaler=self.scaler, fit=False
        )
        self._test_datatuple, _ = em.scale_continuous(
            self.em_dataset, datatuple=test_data, scaler=self.scaler, fit=False
        )

        train_data = DataTupleDataset(
            dataset=self._train_datatuple,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        val_data = DataTupleDataset(
            dataset=self._val_datatuple,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        test_data = DataTupleDataset(
            dataset=self._test_datatuple,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

    @property
    def train_datatuple(self) -> em.DataTuple:
        assert self._train_datatuple is not None
        return self._train_datatuple

    @property
    def val_datatuple(self) -> em.DataTuple:
        assert self._val_datatuple is not None
        return self._val_datatuple

    @property
    def test_datatuple(self) -> em.DataTuple:
        assert self._test_datatuple is not None
        return self._test_datatuple

    @property
    def feature_groups(self) -> dict[str, list[slice]]:
        assert self._feature_groups is not None
        return self._feature_groups

    @property
    def disc_features(self) -> list[str]:
        assert self._disc_features is not None
        return self._disc_features

    @property
    def cont_features(self) -> list[str]:
        assert self._cont_features is not None
        return self._cont_features

    def make_feature_groups(self) -> None:
        """Make feature groups for reconstruction."""
        self._disc_features = self.em_dataset.discrete_features
        self._cont_features = self.em_dataset.continuous_features
        self._feature_groups = dict(
            discrete=self.grouped_features_indexes(self.em_dataset.disc_feature_groups)
        )

    def grouped_features_indexes(self, group_iter: dict[str, list[str]]) -> list[slice]:
        """Group discrete features names according to the first segment of their name.

        Then return a list of their corresponding slices (assumes order is maintained).
        """

        feature_slices = []
        start_idx = 0
        for group in group_iter.values():
            len_group = len(list(group))
            indexes = slice(start_idx, start_idx + len_group)
            feature_slices.append(indexes)
            start_idx += len_group

        return feature_slices
