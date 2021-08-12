from __future__ import annotations
from typing import Union

from kit import implements
import numpy as np
import numpy.typing as npt
from torch.functional import Tensor

from bolts.data.datasets.base import PBDataset
from bolts.data.structures import TargetData
from bolts.transforms.tabular import TabularTransform

__all__ = ["PBTabularDataset"]


class PBTabularDataset(PBDataset):
    def __init__(
        self,
        *,
        x: Tensor | npt.NDArray[Union[np.floating, np.integer]],
        y: TargetData | None = None,
        s: TargetData | None = None,
        transform: TabularTransform | None = None,
        target_transform: TabularTransform | None = None,
        transform_online: bool = False,
        cont_indexes: set[int] | list[int] | None = None,
        disc_indexes: set[int] | list[int] | None = None,
        feature_groups: list[slice] | None = None,
        target_groups: list[slice] | None = None,
    ) -> None:
        super().__init__(x=x, y=y, s=s)
        self.transform = transform
        self.target_transform = target_transform
        self.transform_online = transform_online

        self.cont_indexes = cont_indexes
        self.disc_indexes = disc_indexes
        self.feature_groups = feature_groups
        self.target_groups = target_groups

        if not transform_online:
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(x)

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

    @implements(PBDataset)
    def _sample_x(self, index: int, *, coerce_to_tensor: bool = False) -> Tensor:
        x = super()._sample_x(index, coerce_to_tensor=coerce_to_tensor)
        if self.transform_online and (self.transform is not None):
            x = self.transform(x)
        return x

    def _sample_y(self, index: int) -> Tensor | None:
        y = super()._sample_y(index)
        if (y is not None) and self.transform_online and (self.target_transform is not None):
            y = self.target_transform(y)
        return y
