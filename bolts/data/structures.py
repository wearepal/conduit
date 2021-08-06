"""Data structures."""
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


@dataclass(frozen=True)
class NamedSample:
    x: Tensor | np.ndarray | Image.Image

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def add_field(
        self, y: Tensor | None = None, s: Tensor | None = None, iw: Tensor | None = None
    ) -> NamedSample | BinarySample | BinarySampleIW | TernarySample | TernarySampleIW:
        if y is not None:
            if s is not None:
                if iw is not None:
                    return TernarySampleIW(x=self.x, s=s, y=y, iw=iw)
                return TernarySample(x=self.x, s=s, y=y)
            if iw is not None:
                return BinarySampleIW(x=self.x, y=y, iw=iw)
            return BinarySample(x=self.x, y=y)
        return self


@dataclass(frozen=True)
class BinarySample(NamedSample):
    y: Tensor

    def add_field(
        self, y: Tensor | None = None, s: Tensor | None = None, iw: Tensor | None = None
    ) -> BinarySample | BinarySampleIW | TernarySample | TernarySampleIW:
        if s is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=s, y=self.y, iw=iw)
            return TernarySample(x=self.x, s=s, y=self.y)
        if iw is not None:
            return BinarySampleIW(x=self.x, y=self.y, iw=iw)
        return self


@dataclass(frozen=True)
class BinarySampleIW(BinarySample):
    iw: Tensor

    def add_field(
        self, y: Tensor | None = None, s: Tensor | None = None, iw: Tensor | None = None
    ) -> BinarySampleIW | TernarySampleIW:
        if s is not None:
            return TernarySampleIW(x=self.x, s=s, y=self.y, iw=self.iw)
        return self


@dataclass(frozen=True)
class TernarySample(BinarySample):
    y: Tensor
    s: Tensor

    def add_field(
        self, y: Tensor | None = None, s: Tensor | None = None, iw: Tensor | None = None
    ) -> TernarySample | TernarySampleIW:
        if iw is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=self.y, iw=iw)
        return self


@dataclass(frozen=True)
class TernarySampleIW(TernarySample):
    iw: Tensor

    def add_field(
        self, y: Tensor | None = None, s: Tensor | None = None, iw: Tensor | None = None
    ) -> TernarySampleIW:
        return self


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
