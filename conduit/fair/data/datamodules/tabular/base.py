"""Tabular data-module."""
from abc import abstractmethod
from typing import Dict, List, Optional, Union, cast

import attr
import ethicml as em
from ethicml.data import Dataset, FeatureOrder
from sklearn.preprocessing import StandardScaler
from typing_extensions import final, override

from conduit.data.datamodules import CdtDataModule
from conduit.data.structures import TrainValTestSplit
from conduit.fair.data.datasets import DataTupleDataset

__all__ = ["EthicMlDataModule"]


@attr.define(kw_only=True)
class EthicMlDataModule(CdtDataModule):
    """Base data-module for tabular datasets."""

    scaler: em.ScalerType = attr.field(factory=StandardScaler)
    invert_s: bool = False
    _datatuple: Optional[em.DataTuple] = attr.field(default=None, init=False)
    _train_datatuple: Optional[em.DataTuple] = attr.field(default=None, init=False)
    _val_datatuple: Optional[em.DataTuple] = attr.field(default=None, init=False)
    _test_datatuple: Optional[em.DataTuple] = attr.field(default=None, init=False)
    _cont_features: Optional[List[str]] = attr.field(default=None, init=False)
    _disc_features: Optional[List[str]] = attr.field(default=None, init=False)
    _feature_groups: Optional[Dict[str, Optional[List[slice]]]] = attr.field(
        default=None, init=False
    )

    @property
    @final
    def datatuple(self) -> em.DataTuple:
        self._check_setup_called("datatuple")
        return cast(em.DataTuple, self._datatuple)

    @property
    @abstractmethod
    def em_dataset(self) -> Dataset:
        ...

    @staticmethod
    def _get_split_sizes(train_len: int, *, test_prop: Union[int, float]) -> List[int]:
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

    @override
    def prepare_data(self) -> None:
        self.make_feature_groups()

    @override
    def _get_splits(self) -> TrainValTestSplit[DataTupleDataset]:
        self._datatuple = self.em_dataset.load(order=FeatureOrder.disc_first)

        data_len = int(self._datatuple.x.shape[0])
        num_train_val, num_test = self._get_split_sizes(data_len, test_prop=self.test_prop)
        train_val, test_data = em.train_test_split(
            data=self._datatuple,
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
            self.em_dataset, datatuple=train_data, scaler=self.scaler
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
    def feature_groups(self) -> Dict[str, Optional[List[slice]]]:
        assert self._feature_groups is not None
        return self._feature_groups

    @property
    def disc_features(self) -> List[str]:
        assert self._disc_features is not None
        return self._disc_features

    @property
    def cont_features(self) -> List[str]:
        assert self._cont_features is not None
        return self._cont_features

    def make_feature_groups(self) -> None:
        """Make feature groups for reconstruction."""
        self._disc_features = self.em_dataset.discrete_features
        self._cont_features = self.em_dataset.continuous_features
        assert self.em_dataset.disc_feature_groups is not None
        self._feature_groups = dict(
            discrete=self.grouped_features_indexes(self.em_dataset.disc_feature_groups)
        )

    @staticmethod
    def grouped_features_indexes(group_iter: Dict[str, List[str]]) -> List[slice]:
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
