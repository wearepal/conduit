"""Data structures."""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Union, overload

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
class _SampleBase:
    # Instantiate as NamedSample
    x: Tensor | np.ndarray | Image.Image

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class NamedSample(_SampleBase):
    @overload
    def add_field(self) -> NamedSample:
        ...

    @overload
    def add_field(self, *, y: Tensor) -> BinarySample:
        ...

    @overload
    def add_field(self, *, y: Tensor, iw: Tensor) -> BinarySampleIW:
        ...

    @overload
    def add_field(self, *, y: Tensor, s: Tensor) -> TernarySample:
        ...

    @overload
    def add_field(self, *, y: Tensor, s: Tensor, iw: Tensor) -> TernarySampleIW:
        ...

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
class _BinarySampleMixin:
    y: Tensor


@dataclass(frozen=True)
class BinarySample(_SampleBase, _BinarySampleMixin):
    @overload
    def add_field(self) -> BinarySample:
        ...

    @overload
    def add_field(self, iw: Tensor | None = None) -> BinarySampleIW:
        ...

    @overload
    def add_field(self, s: Tensor | None = None) -> TernarySample:
        ...

    @overload
    def add_field(self, s: Tensor | None = None, iw: Tensor | None = None) -> TernarySampleIW:
        ...

    def add_field(
        self, s: Tensor | None = None, iw: Tensor | None = None
    ) -> BinarySample | BinarySampleIW | TernarySample | TernarySampleIW:
        if s is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=s, y=self.y, iw=iw)
            return TernarySample(x=self.x, s=s, y=self.y)
        if iw is not None:
            return BinarySampleIW(x=self.x, y=self.y, iw=iw)
        return self


@dataclass(frozen=True)
class _IwMixin:
    iw: Tensor


@dataclass(frozen=True)
class BinarySampleIW(_SampleBase, _BinarySampleMixin, _IwMixin):
    @overload
    def add_field(self) -> BinarySampleIW:
        ...

    @overload
    def add_field(self, s: Tensor | None = None) -> TernarySampleIW:
        ...

    def add_field(self, s: Tensor | None = None) -> BinarySampleIW | TernarySampleIW:
        if s is not None:
            return TernarySampleIW(x=self.x, s=s, y=self.y, iw=self.iw)
        return self


@dataclass(frozen=True)
class _TernarySampleMixin:
    s: Tensor


@dataclass(frozen=True)
class TernarySample(_SampleBase, _BinarySampleMixin, _TernarySampleMixin):
    @overload
    def add_field(self) -> TernarySample:
        ...

    @overload
    def add_field(self, iw: Tensor | None = None) -> TernarySampleIW:
        ...

    def add_field(self, iw: Tensor | None = None) -> TernarySample | TernarySampleIW:
        if iw is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=self.y, iw=iw)
        return self


@dataclass(frozen=True)
class TernarySampleIW(_SampleBase, _BinarySampleMixin, _TernarySampleMixin, _IwMixin):
    def add_field(self) -> TernarySampleIW:
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
