from __future__ import annotations
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from torch.functional import Tensor

from bolts.data.datasets.base import PBDataset
from bolts.data.structures import TargetData
from bolts.transforms.tabular import TabularTransform

__all__ = ["PBTabularDataset"]


class PBTabularDataset(PBDataset):

    x: Tensor

    def __init__(
        self,
        *,
        x: Tensor | npt.NDArray[Union[np.floating, np.integer]],
        y: TargetData | None = None,
        s: TargetData | None = None,
        transform: TabularTransform | None = None,
        target_transform: TabularTransform | None = None,
        cont_indexes: list[int] | None = None,
        disc_indexes: list[int] | None = None,
        feature_groups: list[slice] | None = None,
        target_groups: list[slice] | None = None,
    ) -> None:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        super().__init__(x=x, y=y, s=s)
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is not None:
            self.x = self.transform(self.x)
        if self.target_transform is not None:
            self.y = self.target_transform(self.x)

        self.cont_indexes = cont_indexes
        self.disc_indexes = disc_indexes
        self.feature_groups = feature_groups
        self.target_groups = target_groups

    @property
    def x_cont(self) -> Tensor | None:
        if self.cont_indexes is None:
            return None
        return self.x[:, self.cont_indexes]

    @property
    def x_disc(self) -> Tensor | None:
        if self.disc_indexes is None:
            return None
        return self.x[:, self.disc_indexes]
