from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from ranzen import some
import torch
from torch.functional import Tensor

from conduit.data.datasets.base import CdtDataset, I, S, Y
from conduit.data.structures import TargetData
from conduit.transforms.tabular import TabularNormalize

__all__ = ["CdtTabularDataset"]


class CdtTabularDataset(CdtDataset[I, Tensor, Y, S]):
    """A dataset for tabular data.

    :param x: The input features.
    :param y: The target data.
    :param s: The sensitive attribute data.
    :param non_ohe_indexes: The indexes of the non-one-hot-encoded features. If None, all features
        are assumed to be one-hot-encoded. Transformations will only be applied to the
        non-one-hot-encoded features.
    :param feature_groups: List of one-hot-encoded feature groups. Each group is encoded as a slice.
    """

    x: Tensor

    def __init__(
        self,
        *,
        x: Union[Tensor, npt.NDArray[np.floating], npt.NDArray[np.integer]],
        y: Optional[TargetData] = None,
        s: Optional[TargetData] = None,
        non_ohe_indexes: Optional[List[int]] = None,
        feature_groups: Optional[List[slice]] = None,
    ) -> None:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        super().__init__(x=x, y=y, s=s)

        self.non_ohe_indexes = non_ohe_indexes
        self.feature_groups = feature_groups

    @property
    def x_non_ohe(self) -> Optional[Tensor]:
        if self.non_ohe_indexes is None:
            return None
        return self.x[:, self.non_ohe_indexes]

    def fit_transform_(self, transform: TabularNormalize) -> None:
        """Fit a transformation to the non-one-hot-encoded features and transform in-place."""
        if some(x_non_ohe := self.x_non_ohe):
            self.x[:, self.non_ohe_indexes] = transform.fit_transform(x_non_ohe)

    def transform_(self, transform: TabularNormalize) -> None:
        """Transform the non-one-hot-encoded features in-place."""
        if some(x_non_ohe := self.x_non_ohe):
            self.x[:, self.non_ohe_indexes] = transform.transform(x_non_ohe)
