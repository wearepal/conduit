from __future__ import annotations
from typing import NamedTuple

from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Subset

__all__ = ["BinarySample", "TernarySample", "InputSize", "NormalizationValues", "TrainTestSplit"]


class BinarySample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    y: Tensor | float


class TernarySample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    s: Tensor | float
    y: Tensor | float


class InputSize(NamedTuple):
    C: int
    H: int
    W: int


class NormalizationValues(NamedTuple):
    mean: tuple[float, ...]
    std: tuple[float, ...]


class TrainTestSplit(NamedTuple):
    train: Subset
    test: Subset
