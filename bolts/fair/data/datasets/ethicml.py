"""Datasets from EthicML."""
from __future__ import annotations
from itertools import groupby
from typing import Iterator

from ethicml import DataTuple
from ethicml.implementations.pytorch_common import _get_info
import torch

from bolts.data.datasets.base import PBDataset

__all__ = ["DataTupleDataset"]


def group_features(disc_feats: list[str]) -> Iterator[tuple[str, Iterator[str]]]:
    """Group discrete features names according to the first segment of their name."""

    def _first_segment(feature_name: str) -> str:
        return feature_name.split("_")[0]

    return groupby(disc_feats, _first_segment)


def grouped_features_indexes(disc_feats: list[str]) -> list[slice]:
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


class DataTupleDataset(PBDataset):
    """Wrapper for EthicML datasets."""

    def __init__(self, dataset: DataTuple, disc_features: list[str], cont_features: list[str]):
        """Create DataTupleDataset."""
        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = torch.tensor(dataset.x[self.disc_features], dtype=torch.long)
        self.x_cont = torch.tensor(dataset.x[self.cont_features], dtype=torch.long)
        x = torch.cat([self.x_disc, self.x_cont], dim=1)

        (
            _,
            s,
            self.num,
            self.xdim,
            self.sdim,
            self.x_names,
            self.s_names,
        ) = _get_info(dataset)

        y = torch.tensor(dataset.y, dtype=torch.float32)

        self.ydim = dataset.y.shape[1]
        self.y_names = dataset.y.columns

        super().__init__(x=x, y=y, s=s)
