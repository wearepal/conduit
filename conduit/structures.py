from typing import Dict, Union
from typing_extensions import TypeAlias

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR

from .types import METRIC_COLLECTION

__all__ = [
    "LRScheduler",
    "MetricDict",
]


LRScheduler: TypeAlias = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict: TypeAlias = Dict[str, METRIC_COLLECTION]
