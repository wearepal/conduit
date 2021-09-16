from typing import Dict, Union

from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR

__all__ = [
    "LRScheduler",
    "MetricDict",
]


LRScheduler = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict = Dict[str, _METRIC_COLLECTION]
