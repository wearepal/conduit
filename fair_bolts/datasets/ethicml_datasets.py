"""Datasets from EthicML."""
from collections import namedtuple
from itertools import groupby
from typing import Iterator, List, Tuple

import ethicml as em
import numpy as np
import pandas as pd
import torch
from ethicml import DataTuple
from ethicml.implementations.pytorch_common import _get_info
from torch import Tensor
from torch.utils.data import Dataset

DataBatch = namedtuple("DataBatch", ["x", "s", "y", "iw"])


def group_features(disc_feats: List[str]) -> Iterator[Tuple[str, Iterator[str]]]:
    """Group discrete features names according to the first segment of their name."""

    def _first_segment(feature_name: str) -> str:
        return feature_name.split("_")[0]

    return groupby(disc_feats, _first_segment)


def grouped_features_indexes(disc_feats: List[str]) -> List[slice]:
    """Group discrete features names according to the first segment of their name.

    Then return a list of their corresponding slices (assumes order is maintained).
    """
    group_iter = group_features(disc_feats)

    feature_slices = []
    start_idx = 0
    for _, group in group_iter:
        len_group = len(list(group))
        indexes = slice(start_idx, start_idx + len_group)
        feature_slices.append(indexes)
        start_idx += len_group

    return feature_slices


class DataTupleDatasetBase(Dataset):
    """Wrapper for EthicML datasets."""

    def __init__(self, dataset: DataTuple, disc_features: List[str], cont_features: List[str]):
        """Create DataTupleDataset."""
        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = dataset.x[self.disc_features].to_numpy(dtype=np.float32)
        self.x_cont = dataset.x[self.cont_features].to_numpy(dtype=np.float32)

        (
            _,
            self.s,
            self.num,
            self.xdim,
            self.sdim,
            self.x_names,
            self.s_names,
        ) = _get_info(dataset)

        self.y = dataset.y.to_numpy(dtype=np.float32)

        self.ydim = dataset.y.shape[1]
        self.y_names = dataset.y.columns

        dt = em.DataTuple(
            x=pd.DataFrame(
                np.random.randint(0, len(self.s), size=(len(self.s), 1)), columns=list("x")
            ),
            s=pd.DataFrame(self.s, columns=["s"]),
            y=pd.DataFrame(self.y, columns=["y"]),
        )
        self.iws = torch.tensor(em.compute_instance_weights(dt)["instance weights"].values)

    def __len__(self) -> int:
        return self.s.shape[0]

    def _x(self, index: int) -> Tensor:
        x_disc = self.x_disc[index]
        x_cont = self.x_cont[index]
        x = np.concatenate([x_disc, x_cont], axis=0)
        x = torch.from_numpy(x)
        if x.shape == 1:
            x = x.squeeze(0)
        return x

    def _s(self, index: int) -> Tensor:
        s = self.s[index]
        return torch.from_numpy(s).squeeze().long()

    def _y(self, index: int) -> Tensor:
        y = self.y[index]
        return torch.from_numpy(y).squeeze().long()

    def _iw(self, index: int) -> Tensor:
        iw = self.iws[index]
        return iw.squeeze().float()


class DataTupleDataset(DataTupleDatasetBase):
    """Wrapper for EthicML datasets."""

    def __init__(self, dataset: DataTuple, disc_features: List[str], cont_features: List[str]):
        super().__init__(dataset, disc_features, cont_features)

    def __getitem__(self, index: int) -> DataBatch:
        return DataBatch(
            x=self._x(index),
            s=self._s(index).unsqueeze(-1),
            y=self._y(index).unsqueeze(-1),
            iw=self._iw(index).unsqueeze(-1),
        )
