from __future__ import annotations
from typing import Any

import albumentations as A
from torch.utils.data import Dataset

from bolts.data.datasets.utils import ImageTform, apply_image_transform

__all__ = ["ImageTransformer"]


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
