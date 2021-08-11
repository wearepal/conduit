from __future__ import annotations
from enum import Enum
from typing import Dict, Union

from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR

__all__ = [
    "LRScheduler",
    "MetricDict",
    "Stage",
]


class Stage(Enum):
    fit = "fit"
    validate = "validate"
    test = "test"

    def __str__(self):
        return str(self.value)


LRScheduler = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict = Dict[str, _METRIC_COLLECTION]
