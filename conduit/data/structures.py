"""Data structures."""

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast, overload, runtime_checkable
from typing_extensions import Self, override

from PIL import Image
import numpy as np
import numpy.typing as npt
from ranzen.misc import gcopy, reduce_add
from ranzen.types import Addable
import torch
from torch import Tensor

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


@runtime_checkable
class InputContainer(Sized, Addable["InputContainer", "InputContainer"], Protocol):
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
    def __add__(self, other: Self) -> Self: ...

    def to(
        self,
        device: torch.device | str | None,
        *,
        non_blocking: bool = False,
    ) -> Self:
        for name, value in shallow_asdict(self).items():
            if isinstance(value, (Tensor, InputContainer)):
                setattr(self, name, value.to(device, non_blocking=non_blocking))
        return self


RawImage: TypeAlias = npt.NDArray[np.integer[Any]] | Image.Image
UnloadedData: TypeAlias = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | npt.NDArray[np.bytes_] | Tensor
LoadedData: TypeAlias = Tensor | Image.Image | npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | npt.NDArray[np.bytes_] | dict[str, Tensor] | dict[str, Image.Image] | dict[str, npt.NDArray[np.floating[Any]]] | dict[str, npt.NDArray[np.integer[Any]]] | dict[str, npt.NDArray[np.bytes_]] | list[Image.Image] | InputContainer
IndexabledData: TypeAlias = Tensor | npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | npt.NDArray[np.bytes_]

X = TypeVar("X", bound=LoadedData)
X_co = TypeVar("X_co", bound=LoadedData, covariant=True)
XI = TypeVar("XI", bound=IndexabledData)

TargetData: TypeAlias = Tensor | npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]]


def concatenate_inputs(x1: X, x2: X, *, is_batched: bool) -> X:
    if type(x1) != type(x2) or (
        isinstance(x1, list) and type(x1[0]) != type(cast(list[Image.Image], x2)[0])
    ):
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


@dataclass
class MultiCropOutput(InputContainer):
    global_crops: list[Tensor]
    local_crops: list[Tensor] = field(default_factory=list)

    @property
    def all_crops(self) -> list[Tensor]:
        return self.global_crops + self.local_crops

    @property
    def global_crop_sizes(self) -> list[torch.Size]:
        return [crop.shape[-3:] for crop in self.global_crops]

    @property
    def local_crop_sizes(self) -> list[torch.Size]:
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


@dataclass
class SampleBase(InputContainer, Generic[X]):
    x: X

    @override
    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @abstractmethod
    def __iter__(self) -> Iterator[X | Tensor]: ...

    @override
    def __add__(self, other: Self) -> Self:
        return self._get_copy(other, is_batched=True)

    def _get_copy(self, other: Self, is_batched: bool) -> Self:
        copy = gcopy(self, deep=False)
        copy.x = concatenate_inputs(copy.x, other.x, is_batched=is_batched)
        return copy

    def astuple(self, deep: bool = False) -> tuple[X | Tensor, ...]:
        tuple_ = tuple(iter(self))
        if deep:
            tuple_ = gcopy(tuple_, deep=True)
        return tuple_

    def asdict(self, deep: bool = False) -> dict[str, X]:
        if deep:
            asdict(self)
        return shallow_asdict(self)

    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index])


@dataclass
class _BinarySampleMixin:
    y: Tensor

    def _add_to_y(self, other: Self) -> None:
        self.y = torch.cat([self.y, other.y], dim=0)


@dataclass
class _SubgroupSampleMixin:
    s: Tensor

    def _add_to_s(self, other: Self) -> None:
        self.s = torch.cat([self.s, other.s], dim=0)


@dataclass
class _IwMixin:
    iw: Tensor

    def _add_to_iw(self, other: Self) -> None:
        self.iw = torch.cat([self.iw, other.iw], dim=0)


@dataclass
class TernarySampleIW(_IwMixin, _BinarySampleMixin, _SubgroupSampleMixin, SampleBase[X]):
    def add_field(self) -> Self:
        return self

    @override
    def __iter__(self) -> Iterator[X | Tensor]:
        yield from (self.x, self.y, self.s, self.iw)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.y) > 1)
        copy._add_to_y(other)
        copy._add_to_s(other)
        copy._add_to_iw(other)
        return copy

    @override
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(
            self, deep=False, x=self.x[index], y=self.y[index], s=self.s[index], iw=self.iw[index]
        )


@dataclass
class TernarySample(_BinarySampleMixin, _SubgroupSampleMixin, SampleBase[X]):
    @overload
    def add_field(self, iw: None = ...) -> Self: ...

    @overload
    def add_field(self, iw: Tensor) -> TernarySampleIW[X]: ...

    def add_field(self, iw: Tensor | None = None) -> Self | TernarySampleIW[X]:
        if iw is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=self.y, iw=iw)
        return self

    @override
    def __iter__(self) -> Iterator[X | Tensor]:
        yield from (self.x, self.y, self.s)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.y) > 1)
        copy._add_to_y(other)
        copy._add_to_s(other)
        return copy

    @override
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index], y=self.y[index], s=self.s[index])


@dataclass
class BinarySampleIW(_IwMixin, _BinarySampleMixin, SampleBase[X]):
    @overload
    def add_field(self, s: None = ...) -> Self: ...

    @overload
    def add_field(self, s: Tensor) -> TernarySampleIW[X]: ...

    def add_field(self, s: Tensor | None = None) -> Self | TernarySampleIW[X]:
        if s is not None:
            return TernarySampleIW(x=self.x, s=s, y=self.y, iw=self.iw)
        return self

    @override
    def __iter__(self) -> Iterator[X | Tensor]:
        yield from (self.x, self.y, self.iw)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.y) > 1)
        copy._add_to_y(other)
        copy._add_to_iw(other)
        return copy

    @override
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index], y=self.y[index], iw=self.iw[index])


@dataclass
class BinarySample(_BinarySampleMixin, SampleBase[X]):
    @overload
    def add_field(self, *, s: None = ..., iw: None = ...) -> Self: ...

    @overload
    def add_field(self, *, s: None = ..., iw: Tensor) -> BinarySampleIW[X]: ...

    @overload
    def add_field(self, *, s: Tensor, iw: None = ...) -> TernarySample[X]: ...

    @overload
    def add_field(self, *, s: Tensor, iw: Tensor) -> TernarySampleIW[X]: ...

    def add_field(
        self, *, s: Tensor | None = None, iw: Tensor | None = None
    ) -> Self | BinarySampleIW[X] | TernarySample[X] | TernarySampleIW[X]:
        if s is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=s, y=self.y, iw=iw)
            return TernarySample(x=self.x, s=s, y=self.y)
        if iw is not None:
            return BinarySampleIW(x=self.x, y=self.y, iw=iw)
        return self

    @override
    def __iter__(self) -> Iterator[X | Tensor]:
        yield from (self.x, self.y)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.y) > 1)
        copy._add_to_y(other)
        return copy

    @override
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index], y=self.y[index])


@dataclass
class SubgroupSampleIW(SampleBase[X], _SubgroupSampleMixin, _IwMixin):
    @overload
    def add_field(self, y: None = ...) -> Self: ...

    @overload
    def add_field(self, y: Tensor) -> TernarySampleIW[X]: ...

    def add_field(self, y: Tensor | None = None) -> Self | TernarySampleIW[X]:
        if y is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=y, iw=self.iw)
        return self

    @override
    def __iter__(self) -> Iterator[X | Tensor]:
        yield from (self.x, self.s, self.iw)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.s) > 1)
        copy._add_to_s(other)
        copy._add_to_iw(other)
        return copy

    @override
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index], s=self.s[index], iw=self.iw[index])


@dataclass
class SubgroupSample(_SubgroupSampleMixin, SampleBase[X]):
    @overload
    def add_field(self, *, y: None = ..., iw: None = ...) -> Self: ...

    @overload
    def add_field(self, *, y: None = ..., iw: Tensor) -> SubgroupSampleIW[X]: ...

    @overload
    def add_field(self, *, y: Tensor, iw: None = ...) -> TernarySample[X]: ...

    @overload
    def add_field(self, *, y: Tensor, iw: Tensor) -> TernarySampleIW[X]: ...

    def add_field(
        self, *, y: Tensor | None = None, iw: Tensor | None = None
    ) -> Self | SubgroupSampleIW[X] | TernarySample[X] | TernarySampleIW[X]:
        if y is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=self.s, y=y, iw=iw)
            return TernarySample(x=self.x, s=self.s, y=y)
        if iw is not None:
            return SubgroupSampleIW(x=self.x, s=self.s, iw=iw)
        return self

    @override
    def __iter__(self) -> Iterator[X | Tensor]:
        yield from (self.x, self.s)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.s) > 1)
        copy._add_to_s(other)
        return copy

    @override
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index], s=self.s[index])


@dataclass
class NamedSample(SampleBase[X]):
    @overload
    def add_field(self, *, y: None = ..., s: None = ..., iw: None = ...) -> Self: ...

    @overload
    def add_field(self, *, y: Tensor, s: None = ..., iw: None = ...) -> BinarySample[X]: ...

    @overload
    def add_field(self, *, y: Tensor, s: None = ..., iw: Tensor) -> BinarySampleIW[X]: ...

    @overload
    def add_field(self, *, y: Tensor, s: Tensor, iw: None = ...) -> TernarySample[X]: ...

    @overload
    def add_field(self, *, y: Tensor, s: Tensor, iw: Tensor) -> TernarySampleIW[X]: ...

    def add_field(
        self, *, y: Tensor | None = None, s: Tensor | None = None, iw: Tensor | None = None
    ) -> Self | BinarySample[X] | BinarySampleIW[X] | TernarySample[X] | TernarySampleIW[X]:
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
    def __getitem__(self, index: IndexType) -> Self:
        assert isinstance(self.x, (Tensor, np.ndarray)), "x is not indexable"
        return gcopy(self, deep=False, x=self.x[index])


def shallow_astuple(dataclass: object) -> tuple[Any, ...]:
    """dataclasses.astuple() but without the deep-copying/recursion." """
    if not is_dataclass(dataclass):
        raise TypeError("shallow_astuple() should be called on dataclass instances")
    return tuple(getattr(dataclass, field.name) for field in fields(dataclass))


def shallow_asdict(dataclass: object) -> dict[str, Any]:
    """dataclasses.asdict() but without the deep-copying/recursion." """
    if not is_dataclass(dataclass):
        raise TypeError("shallow_asdict() should be called on dataclass instances")
    return {field.name: getattr(dataclass, field.name) for field in fields(dataclass)}


@dataclass
class ImageSize(Sequence[int]):
    c: int
    h: int
    w: int

    def __mul__(self, other: Self | float) -> Self:
        copy = gcopy(self, deep=False)
        if isinstance(other, (float, int)):
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

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        return (self.c, self.h, self.w)[index]

    def __len__(self) -> int:
        return 3

    @property
    def numel(self) -> int:
        return sum(iter(self))


@dataclass(kw_only=True)
class MeanStd:
    mean: tuple[float, ...] | list[float]
    std: tuple[float, ...] | list[float]

    def __iter__(self) -> Iterator[tuple[float, ...] | list[float]]:
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
    def __getitem__(self, index: int) -> R_co: ...


@runtime_checkable
class SizedDataset(Dataset[R_co], Sized, Protocol):
    @override
    def __getitem__(self, index: int) -> R_co: ...

    @override
    def __len__(self) -> int | None:  # type: ignore
        ...


X2 = TypeVar("X2", bound=UnloadedData, covariant=True)
Y = TypeVar("Y", bound=Tensor | None, covariant=True)
S = TypeVar("S", bound=Tensor | None, covariant=True)


@runtime_checkable
class PseudoCdtDataset(Protocol[R_co, X2, Y, S]):
    @property
    def x(self) -> X2: ...
    @property
    def y(self) -> Y: ...
    @property
    def s(self) -> S: ...

    def __getitem__(self, index: int) -> R_co: ...

    def __len__(self) -> int: ...


D = TypeVar("D", bound=Dataset[Any] | Tensor | list[int], covariant=True)


@runtime_checkable
class DatasetWrapper(SizedDataset[R_co], Protocol):
    @property
    def dataset(self) -> Dataset[R_co]: ...

    @override
    def __getitem__(self, index: int) -> R_co: ...

    @override
    def __len__(self) -> int | None:
        if isinstance(self.dataset, SizedDataset):
            return len(self.dataset)  # type: ignore
        return None


@dataclass(kw_only=True)
class TrainTestSplit(Generic[D]):
    train: D
    test: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.test)


@dataclass(kw_only=True)
class TrainValTestSplit(TrainTestSplit[D]):
    val: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.val, self.test)
