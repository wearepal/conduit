from typing import Union

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, StepLR

__all__ = ["LRScheduler"]

LRScheduler = Union[CosineAnnealingWarmRestarts, ExponentialLR, StepLR]
