from typing import Dict, Union

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import TypeAlias

from .types import METRIC_COLLECTION

__all__ = [
    "LRScheduler",
    "MetricDict",
]


LRScheduler: TypeAlias = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict: TypeAlias = Dict[str, METRIC_COLLECTION]
