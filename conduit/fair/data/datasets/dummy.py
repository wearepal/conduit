from typing import Optional, Tuple

import torch

from conduit.data.datasets.base import CdtDataset

__all__ = ["DummyDataset"]


class DummyDataset(CdtDataset):
    """Generate a dummy dataset.

    :param shapes: list of shapes
    :param num_samples: how many samples to use in this dataset
    :param seed: random seed
    """

    def __init__(
        self, *shapes: Tuple[int, ...], num_samples: int = 10000, seed: Optional[int] = None
    ) -> None:
        self.shapes = shapes
        self.num_samples = num_samples

        self.generator = (
            torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
        )

        x = torch.randn(num_samples, *shapes[0], generator=self.generator)
        s = (torch.rand(num_samples, *shapes[1], generator=self.generator) > 0.5).float().squeeze()
        y = (torch.rand(num_samples, *shapes[1], generator=self.generator) > 0.5).float().squeeze()

        super().__init__(x=x, s=s, y=y)
