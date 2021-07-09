from __future__ import annotations
from typing import Any

from torch.utils.data import Dataset

from bolts.data.datasets.utils import (
    ImageTform,
    apply_image_transform,
    compute_instance_weights,
)
from bolts.data.structures import BinarySampleIW, TernarySampleIW

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
        data = self.dataset[index]
        if self.transform is not None:
            data_type = type(data)
            transformed = apply_image_transform(image=data[0], transform=self.transform)
            data = data_type(transformed, *data[1:])
        return data


class InstanceWeightedDataset(Dataset):
    """Wrapper endowing datasets with instance-weights."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.iw = compute_instance_weights(dataset)

    def __getitem__(self, index: int) -> BinarySampleIW | TernarySampleIW:
        sample = self.dataset[index]
        iw = self.iw[index]
        if len(sample) == 2:
            tuple_class = BinarySampleIW
        else:
            tuple_class = TernarySampleIW
        return tuple_class(*sample, iw=iw)

    def __len__(self) -> int:
        return len(self.iw)
