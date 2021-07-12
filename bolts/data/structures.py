from __future__ import annotations
from typing import NamedTuple, Union

import numpy as np
import numpy.typing as npt
from torch import Tensor
from torch.utils.data import Dataset

__all__ = [
    "BinarySample",
    "BinarySampleIW",
    "InputData",
    "InputSize",
    "NormalizationValues",
    "TargetData",
    "TernarySample",
    "TernarySampleIW",
    "TrainTestSplit",
    "TrainValTestSplit",
]


class BinarySample(NamedTuple):
    x: Tensor
    y: Tensor


class BinarySampleIW(NamedTuple):
    x: Tensor
    y: Tensor
    iw: Tensor


class TernarySample(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor


class TernarySampleIW(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor
    iw: Tensor


class InputSize(NamedTuple):
    C: int
    H: int
    W: int


class NormalizationValues(NamedTuple):
    mean: tuple[float, ...]
    std: tuple[float, ...]


class TrainTestSplit(NamedTuple):
    train: Dataset
    test: Dataset


class TrainValTestSplit(NamedTuple):
    train: Dataset
    val: Dataset
    test: Dataset


InputData = npt.NDArray[Union[np.floating, np.integer, np.string_]]
TargetData = Union[Tensor, npt.NDArray[Union[np.floating, np.integer]]]
