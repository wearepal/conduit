from enum import Enum
from typing import Any, Dict, List, Protocol, TypeVar, Union

import numpy as np
import numpy.typing as npt
from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from ranzen.decorators import enum_name_str
from ranzen.torch.loss import ReductionType
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import Protocol, Self, TypeAlias, runtime_checkable

__all__ = [
    "Addable",
    "IndexType",
    "Indexable",
    "LRScheduler",
    "Loss",
    "MetricDict",
    "NDArrayR",
    "Sized",
    "Stage",
]


class Loss(Protocol):
    def __call__(self, input: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        ...

    @property
    def reduction(self) -> Union[ReductionType, str]:
        ...

    @reduction.setter
    def reduction(self, value: Union[ReductionType, str]) -> None:
        ...


@enum_name_str
class Stage(Enum):
    fit = "fit"
    validate = "validate"
    test = "test"


LRScheduler: TypeAlias = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict: TypeAlias = Dict[str, _METRIC_COLLECTION]
NDArrayR: TypeAlias = Union[npt.NDArray[np.floating], npt.NDArray[np.integer]]
IndexType: TypeAlias = Union[int, List[int], slice]

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Sized(Protocol[T_co]):
    def __len__(self) -> int:
        ...


@runtime_checkable
class Addable(Protocol[T_co]):
    def __add__(self, other: Self) -> Self:
        ...


@runtime_checkable
class Indexable(Protocol[T_co]):
    def __getitem__(self, index: IndexType) -> Any:
        ...
