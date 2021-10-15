from typing import Sequence, Union

import numpy as np
from ranzen import implements
import torch
from torch import Tensor

from conduit.data import CdtDataset, RawImage
from conduit.data.datasets.vision.base import CdtVisionDataset


class DummyVisionDataset(CdtVisionDataset):
    def __init__(self, shape: tuple[int, ...], batch_size: int, num_samples: int = 10_000):
        s = torch.randint(2, (100,))
        x = np.array(["foo"] * 100)
        super().__init__(x=x, s=s, y=s, image_dir="")
        self.shape = shape
        self.num_samples = num_samples
        self.batch_size = batch_size

    @implements(CdtDataset)
    def _sample_x(
        self, index: int, *, coerce_to_tensor: bool = False
    ) -> Union[RawImage, Tensor, Sequence[RawImage], Sequence[Tensor]]:
        return torch.rand((3, 32, 32))
