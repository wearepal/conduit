from enum import Enum, auto
from typing import Any, Dict, Union

from kit.decorators import enum_name_str
from kit.torch.loss import ReductionType
from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import Protocol

__all__ = ["LRScheduler", "Loss", "MetricDict", "Stage", "SoundscapeAttr"]


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


@enum_name_str
class SoundscapeAttr(Enum):
    habitat = auto()
    site = auto()
    time = auto()
    NN = auto()
    N0 = auto()


LRScheduler = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict = Dict[str, _METRIC_COLLECTION]
