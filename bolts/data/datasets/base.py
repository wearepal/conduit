from __future__ import annotations
from typing import ClassVar, Optional

from kit import implements
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from bolts.data.structures import BinarySample, InputData, TargetData, TernarySample

__all__ = ["PBDataset"]


class PBDataset(Dataset):
    _repr_indent: ClassVar[int] = 4

    def __init__(
        self, x: InputData, y: Optional[TargetData] = None, s: Optional[TargetData] = None
    ) -> None:
        self.x = x
        if isinstance(y, np.ndarray):
            y = torch.as_tensor(y)
        if isinstance(s, np.ndarray):
            s = torch.as_tensor(s)
        self.y = y
        self.s = s

        self._x_dim = None
        self._y_dim = None
        self._s_dim: int | None = None

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""

    def _sample_x(self, index: int, coerce_to_tensor: bool = False) -> Tensor:
        x = self.x[index]
        if coerce_to_tensor and (not isinstance(x, Tensor)):
            x = torch.as_tensor(x)
        return x

    @property
    def x_dim(
        self,
    ) -> tuple[int, ...]:
        if self._x_dim is None:
            self._x_dim = self._sample_x(0, coerce_to_tensor=True).shape
        return self._x_dim

    @property
    def y_dim(
        self,
    ) -> int | None:
        if (self._y_dim is None) and (self.y is not None):
            self._y_dim = len(self.y.unique())
        return self._y_dim

    @property
    def s_dim(
        self,
    ) -> int | None:
        if (self._s_dim is None) and (self.s is not None):
            self._s_dim = len(self.s.unique())
        return self.s_dim

    @implements(Dataset)
    def __getitem__(self, index: int) -> Tensor | BinarySample | TernarySample:
        data = [self._sample_x(index)]
        if self.y is not None:
            data.append(self.y[index])
        if self.s is not None:
            data.append(self.s[index])
        if len(data) == 2:
            return BinarySample(*data)
        if len(data) == 3:
            return TernarySample(*data)
        return data[0]
