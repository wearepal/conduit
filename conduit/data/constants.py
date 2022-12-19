from typing import Final

from conduit.data.structures import MeanStd

__all__ = ["IMAGENET_STATS"]

IMAGENET_STATS: Final[MeanStd] = MeanStd(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
