"""Dataset wrappers."""
from __future__ import annotations
from dataclasses import is_dataclass, replace
from typing import Any

from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from conduit.data.datasets.utils import (
    ImageTform,
    apply_image_transform,
    compute_instance_weights,
    extract_base_dataset,
)
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import (
    BinarySample,
    BinarySampleIW,
    SampleBase,
    SubgroupSample,
    SubgroupSampleIW,
    TernarySample,
    TernarySampleIW,
    _BinarySampleMixin,
    shallow_asdict,
)
from conduit.transforms.tabular import TabularTransform

__all__ = [
    "ImageTransformer",
    "InstanceWeightedDataset",
    "TabularTransformer",
]


class ImageTransformer(Dataset):
    """
    Wrapper class for applying image transformations.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(self, dataset: Dataset, *, transform: ImageTform | None) -> None:
        self.dataset = dataset
        self._transform: ImageTform | None = None
        self.transform = transform

    def __len__(self) -> int | None:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)  # type: ignore
        return None

    @property
    def transform(self) -> ImageTform | None:
        return self._transform

    @transform.setter
    def transform(self, transform: ImageTform | None) -> None:
        base_dataset = extract_base_dataset(self.dataset, return_subset_indices=False)
        if isinstance(base_dataset, CdtVisionDataset):
            base_dataset.update_il_backend(transform)
        self._transform = transform

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if self.transform is not None:
            if isinstance(sample, (Image.Image, np.ndarray)):
                sample = apply_image_transform(image=sample, transform=self.transform)
            elif isinstance(sample, SampleBase):
                image = sample.x
                assert not isinstance(image, Tensor)
                image = apply_image_transform(image=image, transform=self.transform)
                sample = replace(sample, x=image)
            else:
                image = apply_image_transform(image=sample[0], transform=self.transform)
                data_type = type(sample)
                sample = data_type(image, *sample[1:])
        return sample


class TabularTransformer(Dataset):
    """
    Wrapper class for applying transformations to tabular data.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: TabularTransform | None,
        target_transform: TabularTransform | None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int | None:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)  # type: ignore
        return None

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if self.transform is not None:
            if isinstance(sample, SampleBase):
                new_values = {"x": self.transform(sample.x)}
                if (self.target_transform is not None) and (isinstance(sample, _BinarySampleMixin)):
                    new_values["y"] = self.target_transform(sample.y)
                sample = replace(sample, **new_values)
            elif isinstance(sample, Tensor):
                sample = self.transform(sample)
            else:
                input_ = self.transform(sample[0])
                data_type = type(sample)
                if (self.target_transform is not None) and len(sample) > 1:
                    target = self.target_transform(sample[1])
                    sample = data_type(input_, target, *sample[2:])
                else:
                    sample = data_type(input_, *sample[1:])
        elif self.target_transform is not None:
            if isinstance(sample, _BinarySampleMixin):
                sample = replace(sample, y=self.target_transform(sample.y))
            elif len(sample) > 1:
                data_type = type(sample)
                target = self.target_transform(sample[1])
                sample = data_type(sample[0], target, *sample[2:])
        return sample


class InstanceWeightedDataset(Dataset):
    """Wrapper endowing datasets with instance-weights."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.iw = compute_instance_weights(dataset)

    def __getitem__(self, index: int) -> BinarySampleIW | SubgroupSampleIW | TernarySampleIW:
        sample = self.dataset[index]
        iw = self.iw[index]
        if isinstance(sample, (BinarySample, SubgroupSample, TernarySample)):
            return sample.add_field(iw=iw)
        elif isinstance(sample, tuple):
            if len(sample) == 2:
                x, y = sample
                return BinarySampleIW(x=x, y=y, iw=iw)
            else:
                x, y, s = sample
                return TernarySampleIW(x=x, y=y, s=s, iw=iw)
        elif is_dataclass(sample):
            tuple_class = BinarySampleIW if len(sample) == 2 else TernarySampleIW
            attr_dict = shallow_asdict(sample)
            attr_dict["iw"] = iw  # Covers the corner-case of 'sample' already being an IW sample
            return tuple_class(**attr_dict)
        else:
            raise TypeError(
                f"Sample of type '{type(sample)}` incompatible cannot be converted into an "
                "instance-weighted sample."
            )

    def __len__(self) -> int:
        return len(self.iw)
