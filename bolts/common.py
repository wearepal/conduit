from __future__ import annotations
from typing import Dict, Union

from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import Literal

__all__ = [
    "LRScheduler",
    "MetricDict",
    "Stage",
]


Stage = Literal["fit", "validate", "test"]
LRScheduler = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict = Dict[str, _METRIC_COLLECTION]
