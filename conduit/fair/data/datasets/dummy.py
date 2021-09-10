from __future__ import annotations

import torch

from conduit.data.datasets.base import CdtDataset

__all__ = ["DummyDataset"]


class DummyDataset(CdtDataset):
    """Generate a dummy dataset."""

    def __init__(self, *shapes: tuple[int, ...], num_samples: int = 10000) -> None:
        """
        :param *shapes: list of shapes
        :param num_samples: how many samples to use in this dataset
        """
        self.shapes = shapes
        self.num_samples = num_samples

        x = torch.randn(num_samples, *shapes[0])
        s = (torch.rand(num_samples, *shapes[1]) > 0.5).float().squeeze()
        y = (torch.rand(num_samples, *shapes[1]) > 0.5).float().squeeze()

        super().__init__(x=x, s=s, y=y)
