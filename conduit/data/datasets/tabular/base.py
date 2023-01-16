from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.functional import Tensor

from conduit.data.datasets.base import CdtDataset
from conduit.data.structures import TargetData
from conduit.transforms.tabular import TabularTransform

__all__ = ["CdtTabularDataset"]


class CdtTabularDataset(CdtDataset):
    x: Tensor

    def __init__(
        self,
        *,
        x: Union[Tensor, npt.NDArray[np.floating], npt.NDArray[np.integer]],
        y: Optional[TargetData] = None,
        s: Optional[TargetData] = None,
        transform: Optional[TabularTransform] = None,
        target_transform: Optional[TabularTransform] = None,
        cont_indexes: Optional[List[int]] = None,
        disc_indexes: Optional[List[int]] = None,
        feature_groups: Optional[List[slice]] = None,
        target_groups: Optional[List[slice]] = None,
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
    def x_cont(self) -> Optional[Tensor]:
        if self.cont_indexes is None:
            return None
        return self.x[:, self.cont_indexes]

    @property
    def x_disc(self) -> Optional[Tensor]:
        if self.disc_indexes is None:
            return None
        return self.x[:, self.disc_indexes]
