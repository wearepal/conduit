from collections.abc import Mapping
from enum import auto
from typing import Any, Protocol, TypeAlias, Union, runtime_checkable
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from ranzen import StrEnum
from ranzen.torch.loss import ReductionType
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from torchmetrics import Metric

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

_NUMBER = Union[int, float]  # noqa: UP007
_METRIC = Union[Metric, Tensor, _NUMBER]  # noqa: UP007
METRIC_COLLECTION = Union[_METRIC, Mapping[str, _METRIC]]  # noqa: UP007


class Loss(Protocol):
    reduction: ReductionType | str

    def __call__(self, input: Tensor, target: Tensor, **kwargs: Any) -> Tensor: ...


class Stage(StrEnum):
    FIT = auto()
    VALIDATE = auto()
    TEST = auto()


LRScheduler: TypeAlias = CosineAnnealingWarmRestarts | ExponentialLR | StepLR
MetricDict: TypeAlias = dict[str, METRIC_COLLECTION]
NDArrayR: TypeAlias = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]]
IndexType: TypeAlias = int | list[int] | slice

T_co = TypeVar("T_co", covariant=True, default=Any)


@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...


@runtime_checkable
class Indexable(Protocol[T_co]):
    def __getitem__(self, index: IndexType) -> T_co: ...
