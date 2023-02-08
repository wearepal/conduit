from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from typing_extensions import TypeAlias, override

from conduit.data.structures import TernarySample

from .base import CdtVisionDataset

SampleType: TypeAlias = TernarySample


class DummyVisionDataset(CdtVisionDataset[SampleType, Tensor, Tensor]):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        s_card: Optional[int] = 2,
        y_card: Optional[int] = 2,
        num_samples: int = 10_000,
    ) -> None:
        self.channels = channels
        self.height = height
        self.width = width
        s = torch.randint(s_card, (num_samples,)) if s_card is not None else None
        y = torch.randint(y_card, (num_samples,)) if y_card is not None else None
        x = np.array([""] * num_samples)
        super().__init__(x=x, s=s, y=y, image_dir="")

    @override
    def _load_image(self, index: int) -> npt.NDArray[np.uint8]:
        return np.random.randint(
            0, 256, size=(self.height, self.width, self.channels), dtype="uint8"
        )
