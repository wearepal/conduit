"""Data structures."""
from __future__ import annotations
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


class NamedSample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]


class BinarySample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    y: Tensor

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]


class BinarySampleIW(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    y: Tensor
    iw: Tensor

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]


class TernarySample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    y: Tensor
    s: Tensor

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]


class TernarySampleIW(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    y: Tensor
    s: Tensor
    iw: Tensor

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]


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


InputData = Union[npt.NDArray[Union[np.floating, np.integer, np.string_]], Tensor]
TargetData = Union[Tensor, npt.NDArray[Union[np.floating, np.integer]]]
