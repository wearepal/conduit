"""Data structure wrapper classes."""
from typing import Any, Optional

import albumentations as A
import ethicml as em
import ethicml.vision as emvi
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from torch.utils.data import Dataset

from fair_bolts.datasets.ethicml_datasets import DataBatch


class TiWrapper(Dataset):
    """Wrapper for a Torch Image Datasets."""

    def __init__(self, ti: emvi.TorchImageDataset):
        self.ti = ti
        # Pull out the data components for compatibility with the extract_labels function
        self.x = ti.x
        self.s = ti.s
        self.y = ti.y

        dt = em.DataTuple(
            x=pd.DataFrame(np.random.randint(0, len(ti.s), size=(len(ti.s), 1)), columns=list("x")),
            s=pd.DataFrame(ti.s.cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(ti.y.cpu().numpy(), columns=["y"]),
        )
        self.iws = torch.tensor(em.compute_instance_weights(dt)["instance weights"].values)

    def __getitem__(self, index: int) -> DataBatch:
        x, s, y = self.ti[index]
        iw = self.iws[index].clone().detach()
        return DataBatch(x=x, s=s.long(), y=y.long(), iw=iw.unsqueeze(-1))

    def __len__(self) -> int:
        return len(self.ti)


class AlbumentationsDataset(Dataset):
    """Wrapper class for interfacing with albumentations."""

    def __init__(self, dataset: Dataset, transform: A.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> Optional[int]:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        return None

    def __getitem__(self, index: int) -> Any:
        data = self.dataset[index]
        data_type = type(data)
        if self.transform is not None:
            image = data[0]
            if isinstance(image, Image.Image):
                image = np.array(image)
            # Apply transformations
            augmented = self.transform(image=image)["image"]
            data = data_type(augmented, *data[1:])
        return data
