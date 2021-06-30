from typing import NamedTuple

from torch import Tensor

__all__ = ["DataBatch"]


class DataBatch(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor
    iw: Tensor
