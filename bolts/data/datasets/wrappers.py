from __future__ import annotations
from typing import Any

from PIL.Image import Image
import albumentations as A
import numpy as np
from torch.utils.data import Dataset

__all__ = ["AlbumentationsDataset"]


class AlbumentationsDataset(Dataset):
    """Wrapper class for interfacing between pillow-based datasets and albumentations."""

    def __init__(self, dataset: Dataset, transform: A.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int | None:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)  # type: ignore
        return None

    def __getitem__(self, index: int) -> Any:
        data = self.dataset[index]
        data_type = type(data)
        if self.transform is not None:
            image = data[0]
            if isinstance(image, Image):
                image = np.array(image)
            # Apply transformations
            augmented = self.transform(image=image)["image"]
            data = data_type(augmented, *data[1:])
        return data
