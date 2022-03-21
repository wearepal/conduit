"""Data structures."""
from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from typing import (
    Any,
    Dict,
    Generic,
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
from ranzen.decorators import implements
from ranzen.misc import gcopy
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias

__all__ = [
    "BinarySample",
    "BinarySampleIW",
    "DatasetProt",
    "DatasetWrapper",
    "ImageSize",
    "IndexType",
    "LoadedData",
    "MultiCropOutput",
    "NamedSample",
    "PseudoCdtDataset",
    "RawImage",
    "SampleBase",
    "SubgroupSample",
    "SubgroupSampleIW",
    "TargetData",
    "TernarySample",
    "TernarySampleIW",
    "TrainTestSplit",
    "TrainValTestSplit",
    "UnloadedData",
    "shallow_asdict",
    "shallow_astuple",
]


@dataclass
class MultiCropOutput:
    global_crops: List[Tensor]
    local_crops: List[Tensor] = field(default_factory=list)

    @property
    def all_crops(self) -> List[Tensor]:
        return self.global_crops + self.local_crops

    @property
    def global_crop_sizes(self):
        return [crop.shape[-3:] for crop in self.global_crops]

    @property
    def local_crop_sizes(self):
        return [crop.shape[-3:] for crop in self.local_crops]

    @property
    def shape(self):
        """Shape of the global crops - for compatibility with DMs."""
        return self.global_crops[0].shape

    def __len__(self) -> int:
        """Total number of crops."""
        return len(self.global_crops) + len(self.local_crops)

    def __iadd__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.global_crops += other.global_crops
        copy.local_crops += other.local_crops
        return copy

    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.global_crops = copy.global_crops + other.global_crops
        copy.local_crops += copy.local_crops + other.local_crops
        return copy


RawImage: TypeAlias = Union[npt.NDArray[np.integer], Image.Image]
UnloadedData: TypeAlias = Union[
    npt.NDArray[np.floating],
    npt.NDArray[np.string_],
    Tensor,
]
LoadedData: TypeAlias = Union[
    Tensor,
    RawImage,
    List[RawImage],
    MultiCropOutput,
]
TargetData: TypeAlias = Union[Tensor, npt.NDArray[np.floating], npt.NDArray[np.integer]]

IndexType: TypeAlias = Union[int, List[int], slice]


@dataclass
class SampleBase:
    # Instantiate as NamedSample
    x: LoadedData

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]

    @abstractmethod
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        ...

    def __add__(self, other: Self) -> Self:
        if type(self.x) != type(other.x) or (
            isinstance(self.x, list) and type(self.x[0]) != type(cast(List, other.x)[0])
        ):
            raise AttributeError(
                f"Only {self.__class__.__name__} instances with 'x' attributes of "
                "the same type can be concatenated (added) together."
            )
        copy = gcopy(self, deep=False)
        if isinstance(self.x, (Tensor, np.ndarray)):
            other.x = cast(Union[Tensor, np.ndarray], other.x)
            if self.x.shape != other.x.shape:
                raise AttributeError(
                    f"Only {self.__class__.__name__} instances with 'x' attributes of "
                    "the same shape can be concatenated (added) together: the lhs variable has "
                    f"'x' of shape '{self.x.shape}', the rhs variable 'x' of shape "
                    f"'{other.x.shape}.'"
                )
        if isinstance(copy.x, Tensor):
            other.x = cast(Tensor, other.x)
            copy.x = torch.cat([copy.x, other.x], dim=0)
        elif isinstance(copy.x, np.ndarray):
            other.x = cast(np.ndarray, other.x)
            copy.x = np.concatenate([copy.x, other.x], axis=0)
        elif isinstance(copy.x, Image.Image):
            other.x = cast(Image.Image, other.x)
            copy.x = [copy.x, other.x]
        else:
            copy.x = copy.x + other.x  # type: ignore

        return copy


@dataclass
class NamedSample(SampleBase):
    @overload
    def add_field(self, *, y: None = ..., s: None = ..., iw: None = ...) -> "NamedSample":
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
    ) -> Union["NamedSample", "BinarySample", "BinarySampleIW", "TernarySample", "TernarySampleIW"]:
        if y is not None:
            if s is not None:
                if iw is not None:
                    return TernarySampleIW(x=self.x, s=s, y=y, iw=iw)
                return TernarySample(x=self.x, s=s, y=y)
            if iw is not None:
                return BinarySampleIW(x=self.x, y=y, iw=iw)
            return BinarySample(x=self.x, y=y)
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield self.x


@dataclass
class _BinarySampleMixin:
    y: Tensor


@dataclass
class _SubgroupSampleMixin:
    s: Tensor


@dataclass
class BinarySample(NamedSample, _BinarySampleMixin):
    @overload
    def add_field(self, *, s: None = ..., iw: None = ...) -> "BinarySample":
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
    ) -> Union["BinarySample", "BinarySampleIW", "TernarySample", "TernarySampleIW"]:
        if s is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=s, y=self.y, iw=iw)
            return TernarySample(x=self.x, s=s, y=self.y)
        if iw is not None:
            return BinarySampleIW(x=self.x, y=self.y, iw=iw)
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y)

    @implements(NamedSample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.y = torch.cat(
            [torch.atleast_1d(copy.y), torch.atleast_1d(other.y)],
            dim=0,
        )
        return copy


@dataclass
class SubgroupSample(NamedSample, _SubgroupSampleMixin):
    @overload
    def add_field(self, *, y: None = ..., iw: None = ...) -> "SubgroupSample":
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
    ) -> Union["SubgroupSample", "SubgroupSampleIW", "TernarySample", "TernarySampleIW"]:
        if y is not None:
            if iw is not None:
                return TernarySampleIW(x=self.x, s=self.s, y=y, iw=iw)
            return TernarySample(x=self.x, s=self.s, y=y)
        if iw is not None:
            return SubgroupSampleIW(x=self.x, s=self.s, iw=iw)
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.s)

    @implements(NamedSample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.s = torch.cat(
            [torch.atleast_1d(copy.s), torch.atleast_1d(other.s)],
            dim=0,
        )
        return copy


@dataclass
class _IwMixin:
    iw: Tensor


@dataclass
class BinarySampleIW(BinarySample, _BinarySampleMixin, _IwMixin):
    @overload
    def add_field(self, s: None = ...) -> "BinarySampleIW":
        ...

    @overload
    def add_field(self, s: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(self, s: Optional[Tensor] = None) -> Union["BinarySampleIW", "TernarySampleIW"]:
        if s is not None:
            return TernarySampleIW(x=self.x, s=s, y=self.y, iw=self.iw)
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.iw)

    @implements(BinarySample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.y = torch.cat(
            [torch.atleast_1d(copy.y), torch.atleast_1d(other.y)],
            dim=0,
        )
        return copy


@dataclass
class SubgroupSampleIW(SubgroupSample, _IwMixin):
    @overload
    def add_field(self, y: None = ...) -> "SubgroupSampleIW":
        ...

    @overload
    def add_field(self, y: Tensor = ...) -> "TernarySampleIW":
        ...

    def add_field(self, y: Optional[Tensor] = None) -> Union["SubgroupSampleIW", "TernarySampleIW"]:
        if y is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=y, iw=self.iw)
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.s, self.iw)

    @implements(SubgroupSample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.iw = torch.cat(
            [torch.atleast_1d(copy.iw), torch.atleast_1d(other.iw)],
            dim=0,
        )
        return copy


@dataclass
class TernarySample(BinarySample, _SubgroupSampleMixin):
    @overload
    def add_field(self, iw: None = ...) -> "TernarySample":
        ...

    @overload
    def add_field(self, iw: Tensor) -> "TernarySampleIW":
        ...

    def add_field(self, iw: Optional[Tensor] = None) -> Union["TernarySample", "TernarySampleIW"]:
        if iw is not None:
            return TernarySampleIW(x=self.x, s=self.s, y=self.y, iw=iw)
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.s)

    @implements(BinarySample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.s = torch.cat(
            [torch.atleast_1d(copy.s), torch.atleast_1d(other.s)],
            dim=0,
        )
        return copy


@dataclass
class TernarySampleIW(TernarySample, _IwMixin):
    def add_field(self) -> "TernarySampleIW":
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.s, self.iw)

    @implements(TernarySample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.iw = torch.cat(
            [torch.atleast_1d(copy.iw), torch.atleast_1d(other.iw)],
            dim=0,
        )
        return copy


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
        yield from (self.mean, self.mean)

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
class DatasetProt(Protocol[R_co]):
    def __getitem__(self, index: int) -> R_co:
        ...


@runtime_checkable
class PseudoCdtDataset(Protocol[R_co]):
    x: UnloadedData
    y: Optional[Tensor]
    s: Optional[Tensor]

    def __getitem__(self, index: int) -> R_co:
        ...

    def __len__(self) -> int:
        ...


D = TypeVar("D", bound=DatasetProt)


@runtime_checkable
class DatasetWrapper(Protocol[D]):
    dataset: D

    def __getitem__(self, index: int) -> Any:
        ...


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
