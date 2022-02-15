from enum import Enum
from functools import partial

from torchvision.models import resnet

__all__ = ["ResNetArch"]


class ResNetArch(Enum):
    resnet18 = partial(resnet.resnet18)
    resnet50 = partial(resnet.resnet50)
    resnet34 = partial(resnet.resnet34)
