from enum import Enum, auto
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from ranzen.decorators import enum_name_str
from ranzen.torch.loss import ReductionType
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import Protocol, TypeAlias

__all__ = [
    "LRScheduler",
    "Loss",
    "MetricDict",
    "NDArrayR",
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
