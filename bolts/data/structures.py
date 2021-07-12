from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Union

from PIL import Image
import numpy as np
import numpy.typing as npt
from torch import Tensor
from torch.utils.data import Dataset

__all__ = [
    "BinarySample",
    "BinarySampleIW",
    "InputData",
    "InputSize",
    "NamedSample",
    "NormalizationValues",
    "TargetData",
    "TernarySample",
    "TernarySampleIW",
    "TrainTestSplit",
    "TrainValTestSplit",
]


@dataclass
class NamedSample:
    x: Tensor | np.ndarray | Image.Image


@dataclass
class BinarySample(NamedSample):
    y: Tensor


@dataclass
class BinarySampleIW(BinarySample):
    iw: Tensor


@dataclass
class TernarySample(BinarySample):
    y: Tensor
    s: Tensor


@dataclass
class TernarySampleIW(TernarySample):
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
