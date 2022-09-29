"""Datasets from EthicML."""
from itertools import groupby
from typing import Iterator, List, Tuple

from ethicml import DataTuple
from ethicml.implementations.pytorch_common import _get_info
import torch

from conduit.data.datasets.base import CdtDataset

__all__ = ["DataTupleDataset"]


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


class DataTupleDataset(CdtDataset):
    """Wrapper for EthicML datasets."""

    def __init__(self, dataset: DataTuple, disc_features: List[str], cont_features: List[str]):
        """Create DataTupleDataset."""
        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = torch.as_tensor(dataset.x[self.disc_features].to_numpy(), dtype=torch.long)
        self.x_cont = torch.as_tensor(dataset.x[self.cont_features].to_numpy(), dtype=torch.float)
        x = torch.cat([self.x_disc, self.x_cont], dim=1)

        # NOTE: we should probably not use an internal function from EthicML here
        _, s, self.num, self.xdim, self.x_names, self.s_name = _get_info(dataset)
        self.sdim = 1

        y = torch.as_tensor(dataset.y.to_numpy(), dtype=torch.float32)

        self.ydim = 1
        self.y_name = dataset.y_column

        super().__init__(x=x, y=y, s=s)
