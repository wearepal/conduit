from typing import Dict, Union

from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import TypeAlias

__all__ = [
    "LRScheduler",
    "MetricDict",
]


LRScheduler: TypeAlias = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
MetricDict: TypeAlias = Dict[str, _METRIC_COLLECTION]
