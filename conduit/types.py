from enum import auto
from typing import Any, Dict, List, Mapping, Protocol, TypeVar, Union, runtime_checkable

import numpy as np
import numpy.typing as npt
from ranzen import StrEnum
from ranzen.torch.loss import ReductionType
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from torchmetrics import Metric
from typing_extensions import TypeAlias

__all__ = [
    "IndexType",
    "Indexable",
    "LRScheduler",
    "Loss",
    "MetricDict",
    "NDArrayR",
    "Sized",
    "Stage",
    "METRIC_COLLECTION",
]

_NUMBER = Union[int, float]
_METRIC = Union[Metric, Tensor, _NUMBER]
METRIC_COLLECTION = Union[_METRIC, Mapping[str, _METRIC]]


class Loss(Protocol):
    def __call__(self, input: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        ...

    @property
    def reduction(self) -> Union[ReductionType, str]:
        ...

    @reduction.setter
    def reduction(self, value: Union[ReductionType, str]) -> None:
        ...


class Stage(StrEnum):
    FIT = auto()
    VALIDATE = auto()
    TEST = auto()


LRScheduler: TypeAlias = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict: TypeAlias = Dict[str, METRIC_COLLECTION]
NDArrayR: TypeAlias = Union[npt.NDArray[np.floating], npt.NDArray[np.integer]]
IndexType: TypeAlias = Union[int, List[int], slice]

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Sized(Protocol[T_co]):
    def __len__(self) -> int:
        ...


@runtime_checkable
class Indexable(Protocol[T_co]):
    def __getitem__(self, index: IndexType) -> Any:
        ...
