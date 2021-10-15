from typing import Dict, Union
from typing_extensions import TypeAlias

from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR

__all__ = [
    "LRScheduler",
    "MetricDict",
]


LRScheduler: TypeAlias = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict: TypeAlias = Dict[str, _METRIC_COLLECTION]
