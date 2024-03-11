from typing import TypeAlias

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR

from .types import METRIC_COLLECTION

__all__ = [
    "LRScheduler",
    "MetricDict",
]


LRScheduler: TypeAlias = CosineAnnealingWarmRestarts | ExponentialLR | StepLR
MetricDict: TypeAlias = dict[str, METRIC_COLLECTION]
