"""Utility functions."""
from __future__ import annotations
from functools import lru_cache
from typing import Union, cast

import ethicml.vision as emvi
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset

from bolts.data import AlbumentationsDataset
from bolts.fair.data.datasets import TiWrapper

__all__ = ["extract_labels_from_dataset"]

_Dataset = Union[emvi.TorchImageDataset, TiWrapper]
ExtractableDataset = Union[ConcatDataset[_Dataset], _Dataset, AlbumentationsDataset]


@lru_cache(typed=True)
def extract_labels_from_dataset(dataset: ExtractableDataset) -> tuple[Tensor, Tensor]:
    """Extract labels from a dataset."""

    def _extract(dataset: _Dataset) -> tuple[Tensor, Tensor]:
        if isinstance(dataset, Subset):
            _s = cast(Tensor, dataset.dataset.s[dataset.indices])
            _y = cast(Tensor, dataset.dataset.y[dataset.indices])
        else:
            _s = dataset.s
            _y = dataset.y

        _s = torch.from_numpy(_s) if isinstance(_s, np.ndarray) else _s
        _y = torch.from_numpy(_y) if isinstance(_y, np.ndarray) else _y
        return _s, _y

    try:
        if isinstance(dataset, AlbumentationsDataset):
            dataset = dataset.dataset
        if isinstance(dataset, (ConcatDataset)):
            s_all_ls, y_all_ls = [], []
            for _dataset in dataset.datasets:
                s, y = _extract(_dataset)
                s_all_ls.append(s)
                y_all_ls.append(y)
            s_all = torch.cat(s_all_ls, dim=0)
            y_all = torch.cat(y_all_ls, dim=0)
        else:
            s_all, y_all = _extract(dataset)
    except AttributeError:
        # Resort to the Brute-force approach of iterating over the dataset
        s_all_ls, y_all_ls = [], []
        for batch in dataset:
            s_all_ls.append(batch[1])
            y_all_ls.append(batch[2])
        s_all = torch.cat(s_all_ls, dim=0)
        y_all = torch.cat(y_all_ls, dim=0)
    return s_all, y_all
