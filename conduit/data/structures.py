"""Data structures."""
from __future__ import annotations
from abc import abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from PIL import Image
import attr
import numpy as np
import numpy.typing as npt
from ranzen.misc import gcopy, reduce_add
from ranzen.types import Addable
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias, override

from conduit.types import IndexType, Sized

__all__ = [
    "BinarySample",
    "BinarySampleIW",
    "Dataset",
    "DatasetWrapper",
    "ImageSize",
    "InputContainer",
    "LoadedData",
    "MeanStd",
    "MultiCropOutput",
    "NamedSample",
    "PseudoCdtDataset",
    "RawImage",
    "SampleBase",
    "SizedDataset",
    "SubgroupSample",
    "SubgroupSampleIW",
    "TargetData",
    "TernarySample",
    "TernarySampleIW",
    "TrainTestSplit",
    "TrainValTestSplit",
    "UnloadedData",
    "concatenate_inputs",
    "shallow_asdict",
    "shallow_astuple",
]

RawImage: TypeAlias = Union[npt.NDArray[np.integer], Image.Image]
UnloadedData: TypeAlias = Union[
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    npt.NDArray[np.string_],
    Tensor,
]
LoadedData: TypeAlias = Union[
    Tensor,
    Image.Image,
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    npt.NDArray[np.string_],
    Dict[str, Tensor],
    Dict[str, Image.Image],
    Dict[str, npt.NDArray[np.floating]],
    Dict[str, npt.NDArray[np.integer]],
    Dict[str, npt.NDArray[np.string_]],
    List[Image.Image],
    "InputContainer[X_co]",
]
IndexabledData: TypeAlias = Union[
    Tensor,
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    npt.NDArray[np.string_],
]

X = TypeVar("X", bound=LoadedData)
X_co = TypeVar("X_co", bound=LoadedData, covariant=True)
XI = TypeVar("XI", bound=IndexabledData)

TargetData: TypeAlias = Union[Tensor, npt.NDArray[np.floating], npt.NDArray[np.integer]]


def concatenate_inputs(x1: X, x2: X, *, is_batched: bool) -> X:
    if type(x1) != type(x2) or (isinstance(x1, list) and type(x1[0]) != type(cast(List, x2)[0])):
        raise AttributeError("Only data of the same type can be concatenated (added) together.")
    if isinstance(x1, Tensor):
        # if the number of dimensions is different by 1, append a batch dimension.
        ndim_diff = x1.ndim - x2.ndim  # type: ignore
        if ndim_diff == 1:
            x2 = x2.unsqueeze(0)  # type: ignore
        elif ndim_diff == -1:
            x1 = x1.unsqueeze(0)
        if is_batched:
            return torch.cat([x1, x2], dim=0)  # type: ignore
        return torch.stack([x1, x2], dim=0)  # type: ignore

    elif isinstance(x1, np.ndarray):
        # if the number of dimensions is different by 1, append a batch dimension.
        ndim_diff = x1.ndim - x2.ndim  # type: ignore
        if ndim_diff == 1:
            x2 = np.expand_dims(x2, axis=0)  # type: ignore
        elif ndim_diff == -1:
            x1 = np.expand_dims(x1, axis=0)  # type: ignore
        if is_batched:
            return np.concatenate([x1, x2], axis=0)  # type: ignore
        return np.stack([x1, x2], axis=0)  # type: ignore
    elif isinstance(x1, Image.Image):
        return [x1, x2]  # type: ignore
    elif isinstance(x1, dict):
        for key, value in x2.items():  # type: ignore
            if key in x1:
                x1[key] = concatenate_inputs(x1[key], value, is_batched=is_batched)  # type: ignore
            else:
                x1[key] = value  # type: ignore
            return x1
    return x1 + x2  # type: ignore


@runtime_checkable
class InputContainer(Sized, Addable, Protocol[X_co]):
    @classmethod
    def fromiter(cls, sequence: Iterable[Self]) -> Self:
        """
        Collates a sequence of container instances into a single instance.

        :param sequence: Sequence of containers to be collated.

        :returns: A collated container.
        """
        return reduce_add(sequence)

    @override
    def __len__(self) -> int:
        """Total number of samples in the container."""
        ...

    @override
    def __add__(self, other: Self) -> Self:
        ...

    def to(
        self,
        device: Optional[Union[torch.device, str]],
        *,
        non_blocking: bool = False,
    ) -> Self:
        for name, value in shallow_asdict(self).items():
            if isinstance(value, (Tensor, InputContainer)):
                setattr(self, name, value.to(device, non_blocking=non_blocking))
        return self


@dataclass
class MultiCropOutput(InputContainer[Tensor]):
    global_crops: List[Tensor]
    local_crops: List[Tensor] = field(default_factory=list)

    @property
    def all_crops(self) -> List[Tensor]:
        return self.global_crops + self.local_crops

    @property
    def global_crop_sizes(self) -> List[torch.Size]:
        return [crop.shape[-3:] for crop in self.global_crops]

    @property
    def local_crop_sizes(self) -> List[torch.Size]:
        return [crop.shape[-3:] for crop in self.local_crops]

    @property
    def shape(self) -> torch.Size:
        """Shape of the global crops - for compatibility with DMs."""
        return self.global_crops[0].shape

    @override
    def __len__(self) -> int:
        """Total number of crops."""
        return len(self.global_crops) + len(self.local_crops)

    def __iadd__(self, other: Self) -> Self:
        self.global_crops += other.global_crops
        self.local_crops += other.local_crops
        return self

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.global_crops = copy.global_crops + other.global_crops
        copy.local_crops = copy.local_crops + other.local_crops
        return copy


IS = TypeVar("IS", bound="SampleBase[IndexabledData]")


@dataclass
class SampleBase(InputContainer[X]):
    x: X

    @override
    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @abstractmethod
    def __iter__(self) -> Iterator[X]:
        ...

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.x = concatenate_inputs(copy.x, other.x, is_batched=True)
        return copy

    def astuple(self, deep: bool = False) -> Tuple[X]:
        tuple_ = tuple(iter(self))
        if deep:
            tuple_ = gcopy(tuple_, deep=True)
        return tuple_

    def asdict(self, deep: bool = False) -> Dict[str, X]:
        if deep:
            asdict(self)
        return shallow_asdict(self)

    def __getitem__(self: "SampleBase[XI]", index: IndexType) -> "SampleBase[XI]":
        return gcopy(self, deep=False, x=self.x[index])


@dataclass
class NamedSample(SampleBase[X]):
    @overload
    def add_field(self, *, y: None = ..., s: None = ..., iw: None = ...) -> Self:
        ...

    @overload
    def add_field(self, *, y: Tensor = ..., s: None = ..., iw: None = ...) -> "BinarySample":
        ...

    @overload
    def add_field(self, *, y: Tensor = ..., s: None = ..., iw: Tensor = ...) -> "BinarySampleIW":
        ...

    @overload
    def add_field(self, *, y: Tensor = ..., s: Tensor = ..., iw: None = ...) -> "TernarySample":
        ...

    @overload
    def add_field(self, *, y: Tensor = ..., s: Tensor = ..., iw: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(
        self, y: Optional[Tensor] = None, s: Optional[Tensor] = None, iw: Optional[Tensor] = None
    ) -> Union[Self, "BinarySample", "BinarySampleIW", "TernarySample", "TernarySampleIW"]:
        if y is not None:
            if s is not None:
                if iw is not None:
                    return TernarySampleIW(x=self.x, s=s, y=y, iw=iw)
                return TernarySample(x=self.x, s=s, y=y)
            if iw is not None:
                return BinarySampleIW(x=self.x, y=y, iw=iw)
            return BinarySample(x=self.x, y=y)
        return self

    @override
    def __iter__(self) -> Iterator[X]:
        yield self.x

    @override
    def __getitem__(self: "NamedSample[XI]", index: IndexType) -> "NamedSample[XI]":
        return gcopy(self, deep=False, x=self.x[index])


@dataclass
class _BinarySampleMixin:
    y: Tensor


@dataclass
class _SubgroupSampleMixin:
    s: Tensor


@dataclass
class BinarySample(NamedSample[X], _BinarySampleMixin):
    @overload
    def add_field(self, *, s: None = ..., iw: None = ...) -> Self:
        ...

    @overload
    def add_field(self, *, s: None = ..., iw: Tensor = ...) -> "BinarySampleIW":
        ...

    @overload
    def add_field(self, *, s: Tensor = ..., iw: None = ...) -> "TernarySample":
        ...

    @overload
    def add_field(self, *, s: Tensor = ..., iw: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(
        self, *, s: Optional[Tensor] = None, iw: Optional[Tensor] = None
    ) -> Union[Self, "BinarySampleIW", "TernarySample", "TernarySampleIW"]:
        if s is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=s, y=self.y, iw=iw)
            return TernarySample(x=self.x, s=s, y=self.y)
        if iw is not None:
            return BinarySampleIW(x=self.x, y=self.y, iw=iw)
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:  # type: ignore
        yield from (self.x, self.y)

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.y = torch.cat([copy.y, other.y], dim=0)
        copy.x = concatenate_inputs(copy.x, other.x, is_batched=len(copy.y) > 1)
        return copy

    @override
    def __getitem__(self: "BinarySample[XI]", index: IndexType) -> "BinarySample[XI]":
        return gcopy(self, deep=False, x=self.x[index], y=self.y[index])


@dataclass
class SubgroupSample(NamedSample[X], _SubgroupSampleMixin):
    @overload
    def add_field(self, *, y: None = ..., iw: None = ...) -> Self:
        ...

    @overload
    def add_field(self, *, y: None = ..., iw: Tensor = ...) -> "SubgroupSampleIW":
        ...

    @overload
    def add_field(self, *, y: Tensor = ..., iw: None = ...) -> "TernarySample":
        ...

    @overload
    def add_field(self, *, y: Tensor = ..., iw: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(
        self, *, y: Optional[Tensor] = None, iw: Optional[Tensor] = None
    ) -> Union[Self, "SubgroupSampleIW", "TernarySample", "TernarySampleIW"]:
        if y is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=self.s, y=y, iw=iw)
            return TernarySample(x=self.x, s=self.s, y=y)
        if iw is not None:
            return SubgroupSampleIW(x=self.x, s=self.s, iw=iw)
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:  # type: ignore
        yield from (self.x, self.s)

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.s = torch.cat([copy.s, other.s], dim=0)
        copy.x = concatenate_inputs(copy.x, other.x, is_batched=len(copy.s) > 1)
        return copy

    @override
    def __getitem__(self: "SubgroupSample[XI]", index: IndexType) -> "SubgroupSample[XI]":
        return gcopy(self, deep=False, x=self.x[index], s=self.s[index])


@dataclass
class _IwMixin:
    iw: Tensor


@dataclass
class BinarySampleIW(BinarySample[X], _BinarySampleMixin, _IwMixin):
    @overload
    def add_field(self, s: None = ...) -> Self:
        ...

    @overload
    def add_field(self, s: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(self, s: Optional[Tensor] = None) -> Union[Self, "TernarySampleIW"]:
        if s is not None:
            return TernarySampleIW(x=self.x, s=s, y=self.y, iw=self.iw)
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.iw)

    @override
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.iw = torch.cat([copy.iw, other.iw], dim=0)
        return copy

    @override
    def __getitem__(self: "BinarySampleIW[XI]", index: IndexType) -> "BinarySampleIW[XI]":
        return gcopy(self, deep=False, x=self.x[index], y=self.y[index], iw=self.iw[index])


@dataclass
class SubgroupSampleIW(SubgroupSample[X], _IwMixin):
    @overload
    def add_field(self, y: None = ...) -> Self:
        ...

    @overload
    def add_field(self, y: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(self, y: Optional[Tensor] = None) -> Union[Self, "TernarySampleIW"]:
        if y is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=y, iw=self.iw)
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.s, self.iw)

    @override
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.iw = torch.cat([copy.iw, other.iw], dim=0)
        return copy

    @override
    def __getitem__(self: "SubgroupSampleIW[XI]", index: IndexType) -> "SubgroupSampleIW[XI]":
        return gcopy(self, deep=False, x=self.x[index], s=self.s[index], iw=self.iw[index])


@dataclass
class TernarySample(BinarySample[X], _SubgroupSampleMixin):
    @overload
    def add_field(self, iw: None = ...) -> Self:
        ...

    @overload
    def add_field(self, iw: Tensor) -> Self:
        ...

    def add_field(self, iw: Optional[Tensor] = None) -> Union[Self, "TernarySampleIW"]:
        if iw is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=self.y, iw=iw)
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.s)

    @override
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.s = torch.cat([copy.s, other.s], dim=0)
        return copy

    @override
    def __getitem__(self: "TernarySample[XI]", index: IndexType) -> "TernarySample[XI]":
        return gcopy(self, deep=False, x=self.x[index], y=self.y[index], s=self.s[index])


@dataclass
class TernarySampleIW(TernarySample[X], _IwMixin):
    def add_field(self) -> Self:
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.s, self.iw)

    @override
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.iw = torch.cat([copy.iw, other.iw], dim=0)
        return copy

    @override
    def __getitem__(self: "TernarySampleIW[XI]", index: IndexType) -> "TernarySampleIW[XI]":
        return gcopy(
            self, deep=False, x=self.x[index], y=self.y[index], s=self.s[index], iw=self.iw[index]
        )


def shallow_astuple(dataclass: object) -> Tuple[Any, ...]:
    """dataclasses.astuple() but without the deep-copying/recursion." """
    if not is_dataclass(dataclass):
        raise TypeError("shallow_astuple() should be called on dataclass instances")
    return tuple(getattr(dataclass, field.name) for field in fields(dataclass))


def shallow_asdict(dataclass: object) -> Dict[str, Any]:
    """dataclasses.asdict() but without the deep-copying/recursion." """
    if not is_dataclass(dataclass):
        raise TypeError("shallow_asdict() should be called on dataclass instances")
    return {field.name: getattr(dataclass, field.name) for field in fields(dataclass)}


@attr.define
class ImageSize:
    c: int
    h: int
    w: int

    def __mul__(self, other: Union[Self, float]) -> Self:
        copy = gcopy(self, deep=False)
        if isinstance(other, float):
            copy.c = round(copy.c * other)
            copy.h = round(copy.h * other)
            copy.w = round(copy.w * other)
        else:
            copy.c *= other.c
            copy.h *= other.h
            copy.w *= other.w
        return copy

    def __iter__(self) -> Iterator[int]:
        yield from (self.c, self.h, self.w)

    @property
    def numel(self) -> int:
        return sum(iter(self))


@attr.define(kw_only=True)
class MeanStd:
    mean: Union[Tuple[float, ...], List[float]]
    std: Union[Tuple[float, ...], List[float]]

    def __iter__(self) -> Iterator[Union[Tuple[float, ...], List[float]]]:
        yield from (self.mean, self.std)

    def __imul__(self, value: float) -> Self:
        self.mean = [value * elem for elem in self.mean]
        self.std = [value * elem for elem in self.std]
        return self

    def __mul__(self, value: float) -> Self:
        copy = gcopy(self, deep=True)
        copy *= value
        return copy

    def __idiv__(self, value: float) -> Self:
        self *= 1 / value
        return self

    def __div__(self, value: float) -> Self:
        copy = gcopy(self, deep=True)
        copy *= 1 / value
        return copy


R_co = TypeVar("R_co", covariant=True)


@runtime_checkable
class Dataset(Protocol[R_co]):
    def __getitem__(self, index: int) -> R_co:
        ...


@runtime_checkable
class SizedDataset(Dataset[R_co], Sized, Protocol):
    @override
    def __getitem__(self, index: int) -> R_co:
        ...

    @override
    def __len__(self) -> Optional[int]:
        ...


X2 = TypeVar("X2", bound=UnloadedData)
Y = TypeVar("Y", Tensor, None)
S = TypeVar("S", Tensor, None)


@runtime_checkable
class PseudoCdtDataset(Protocol[R_co, X2, Y, S]):
    x: X2
    y: Y
    s: S

    def __getitem__(self, index: int) -> R_co:
        ...

    def __len__(self) -> int:
        ...


D = TypeVar("D", bound=Union[Dataset, Tensor, List[int]])


@runtime_checkable
class DatasetWrapper(SizedDataset[R_co], Protocol):
    dataset: Dataset

    @override
    def __getitem__(self, index: int) -> R_co:
        ...

    @override
    def __len__(self) -> Optional[int]:
        if isinstance(self.dataset, SizedDataset):
            return len(self.dataset)
        return None


@attr.define(kw_only=True)
class TrainTestSplit(Generic[D]):
    train: D
    test: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.test)


@attr.define(kw_only=True)
class TrainValTestSplit(TrainTestSplit[D]):
    val: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.val, self.test)
