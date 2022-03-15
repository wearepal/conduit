"""Data structures."""
from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from PIL import Image
import attr
import numpy as np
import numpy.typing as npt
from ranzen.decorators import implements
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypeAlias

__all__ = [
    "BinarySample",
    "BinarySampleIW",
    "DatasetProt",
    "ImageSize",
    "InputData",
    "MultiCropOutput",
    "NamedSample",
    "PseudoCdtDataset",
    "SampleBase",
    "SubgroupSample",
    "SubgroupSampleIW",
    "TargetData",
    "TernarySample",
    "TernarySampleIW",
    "TrainTestSplit",
    "TrainValTestSplit",
    "shallow_asdict",
    "shallow_astuple",
]


InputData: TypeAlias = Union[
    npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.string_], Tensor
]
TargetData: TypeAlias = Union[Tensor, npt.NDArray[np.floating], npt.NDArray[np.integer]]


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


@dataclass
class SampleBase:
    # Instantiate as NamedSample
    x: Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]

    @abstractmethod
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        ...


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
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
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
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        yield from (self.x, self.y)


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
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        yield from (self.x, self.s)


@dataclass
class _IwMixin:
    iw: Tensor


@dataclass
class BinarySampleIW(SampleBase, _BinarySampleMixin, _IwMixin):
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
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        yield from (self.x, self.y, self.iw)


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
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        yield from (self.x, self.s, self.iw)


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
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        yield from (self.x, self.y, self.s)


@dataclass
class TernarySampleIW(TernarySample, _IwMixin):
    def add_field(self) -> "TernarySampleIW":
        return self

    @implements(SampleBase)
    def __iter__(self) -> Iterator[Union[Tensor, np.ndarray, Image.Image, MultiCropOutput]]:
        yield from (self.x, self.y, self.s, self.iw)


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


class ImageSize(NamedTuple):
    C: int
    H: int
    W: int


class MeanStd(NamedTuple):
    mean: Union[Tuple[float, ...], List[float]]
    std: Union[Tuple[float, ...], List[float]]


@runtime_checkable
class DatasetProt(Protocol):
    def __getitem__(self, index: int) -> Any:
        ...


@runtime_checkable
class PseudoCdtDataset(Protocol):
    x: InputData
    y: Optional[Tensor]
    s: Optional[Tensor]

    def __getitem__(self, index: int) -> Any:
        ...


D = TypeVar("D", Dataset, PseudoCdtDataset, covariant=True)


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
