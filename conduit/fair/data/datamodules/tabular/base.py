"""Tabular data-module."""
from __future__ import annotations
from abc import abstractmethod

import attr
import ethicml as em
from ethicml import DataTuple
from ethicml.preprocessing.scaling import ScalerType
from kit import implements
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from typing_extensions import final

from conduit.data.datamodules import CdtDataModule
from conduit.data.structures import TrainValTestSplit
from conduit.fair.data.datasets import DataTupleDataset

__all__ = ["EthicMlDataModule"]


@attr.define(kw_only=True)
class EthicMlDataModule(CdtDataModule):
    """Base data-module for tabular datasets."""

    scaler: ScalerType = attr.field(default=StandardScaler())
    _datatuple: DataTuple = attr.field(default=None, init=False)
    _train_datatuple: em.DataTuple | None = attr.field(default=None, init=False)
    _val_datatuple: em.DataTuple | None = attr.field(default=None, init=False)
    _test_datatuple: em.DataTuple | None = attr.field(default=None, init=False)
    _cont_features: list[str] | None = attr.field(default=None, init=False)
    _disc_features: list[str] | None = attr.field(default=None, init=False)
    _feature_groups: dict[str, list[slice]] | None = attr.field(default=None, init=False)

    @property
    @final
    def datatuple(self) -> DataTuple:
        if self._datatuple is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.datatuple' cannot be accessed as '{cls_name}.setup' has "
                "not yet been called.'"
            )
        return self._datatuple

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
        self.make_feature_groups()

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        self._datatuple = self.em_dataset.load(ordered=True)

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

    @staticmethod
    def grouped_features_indexes(group_iter: dict[str, list[str]]) -> list[slice]:
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
