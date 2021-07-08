from typing import Union

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR
from typing_extensions import Literal

__all__ = ["LRScheduler", "SchedInterval"]
SchedInterval = Literal["epoch", "step"]
LRScheduler = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
