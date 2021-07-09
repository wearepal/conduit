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
        self.y = y
        self.s = s

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

    def _sample_x(self, index: int) -> Tensor:
        x = self.x[index]
        if not isinstance(x, Tensor):
            x = torch.as_tensor(x)
        return x

    @implements(Dataset)
    def __getitem__(self, index: int) -> Tensor | BinarySample | TernarySample:
        data = [self._sample_x(index)]
        if self.y is not None:
            data.append(self.y[index])
        if self.y is not None:
            data.append(self.y[index])
        if len(data) == 2:
            return BinarySample(*data)
        if len(data) == 1:
            return TernarySample(*data)
        return data[0]
