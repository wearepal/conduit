from dataclasses import replace
from typing import Any, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from conduit.data.datasets.utils import extract_base_dataset
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.datasets.vision.utils import ImageTform, apply_image_transform
from conduit.data.structures import Dataset, DatasetWrapper, RawImage, SampleBase

__all__ = [
    "ImageTransformer",
]


class ImageTransformer(DatasetWrapper[Any]):
    """
    Wrapper class for applying image transformations.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(self, dataset: Dataset, *, transform: Optional[ImageTform]) -> None:
        self.dataset = dataset
        self._transform: Optional[ImageTform] = None
        self.transform = transform

    @property
    def transform(self) -> Optional[ImageTform]:
        return self._transform

    @transform.setter
    def transform(self, transform: Optional[ImageTform]) -> None:
        base_dataset = extract_base_dataset(self.dataset, return_subset_indices=False)
        if isinstance(base_dataset, CdtVisionDataset):
            base_dataset.update_il_backend(transform)
        self._transform = transform

    @override
    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]

        def _transform_sample(
            _sample: Union[RawImage, SampleBase, Tuple[Union[RawImage, Image.Image], ...]]
        ) -> Any:
            if self.transform is None:
                return _sample
            if isinstance(_sample, (Image.Image, np.ndarray)):
                return apply_image_transform(image=_sample, transform=self.transform)
            elif isinstance(_sample, SampleBase):
                image = _sample.x
                if isinstance(image, list):
                    image = [_transform_sample(elem) for elem in image]
                    if isinstance(image[0], Tensor):
                        image = torch.stack(image, dim=0)
                    elif isinstance(image[0], np.ndarray):
                        image = np.stack(image, axis=0)

                elif not isinstance(image, (Image.Image, np.ndarray)):
                    raise TypeError(
                        f"Image transform cannot be applied to input of type '{type(image)}'"
                        "(must be a PIL Image or a numpy array)."
                    )
                else:
                    image = apply_image_transform(image=image, transform=self.transform)
                return replace(_sample, x=image)
            else:
                image = apply_image_transform(image=_sample[0], transform=self.transform)
                data_type = type(_sample)
                return data_type(image, *_sample[1:])  # type: ignore

        return _transform_sample(sample)
