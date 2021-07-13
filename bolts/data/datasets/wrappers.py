from __future__ import annotations
from dataclasses import replace
from typing import Any

from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from bolts.data.datasets.utils import (
    ImageTform,
    apply_image_transform,
    compute_instance_weights,
)
from bolts.data.structures import BinarySampleIW, NamedSample, TernarySampleIW

__all__ = ["ImageTransformer", "InstanceWeightedDataset"]


class ImageTransformer(Dataset):
    """
    Wrapper class for applying image transformations.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(self, dataset: Dataset, transform: ImageTform | None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int | None:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)  # type: ignore
        return None

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if self.transform is not None:
            if isinstance(sample, (Image.Image, np.ndarray)):
                sample = apply_image_transform(image=sample, transform=self.transform)
            elif isinstance(sample, NamedSample):
                image = sample.x
                assert not isinstance(image, Tensor)
                image = apply_image_transform(image=image, transform=self.transform)
                sample = replace(sample, x=image)
            else:
                image = apply_image_transform(image=sample[0], transform=self.transform)
                data_type = type(sample)
                sample = data_type(image, *sample[1:])
        return sample


class InstanceWeightedDataset(Dataset):
    """Wrapper endowing datasets with instance-weights."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.iw = compute_instance_weights(dataset)

    def __getitem__(self, index: int) -> BinarySampleIW | TernarySampleIW:
        sample = self.dataset[index]
        iw = self.iw[index]
        tuple_class = BinarySampleIW if len(sample) == 2 else TernarySampleIW
        return tuple_class(*sample, iw=iw)

    def __len__(self) -> int:
        return len(self.iw)
