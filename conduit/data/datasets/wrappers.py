"""Dataset wrappers."""
from dataclasses import is_dataclass, replace
from typing import Any, Optional, Tuple, Union

from PIL import Image
import numpy as np
from ranzen.decorators import implements
import torch
from torch import Tensor

from conduit.data.datasets.utils import (
    AudioTform,
    ImageTform,
    apply_audio_transform,
    apply_image_transform,
    compute_instance_weights,
    extract_base_dataset,
)
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import (
    BinarySample,
    BinarySampleIW,
    Dataset,
    DatasetWrapper,
    NamedSample,
    RawImage,
    SampleBase,
    SubgroupSample,
    TernarySample,
    TernarySampleIW,
    _BinarySampleMixin,
    shallow_asdict,
)
from conduit.transforms.tabular import TabularTransform

__all__ = [
    "AudioTransformer",
    "ImageTransformer",
    "InstanceWeightedDataset",
    "TabularTransformer",
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

    @implements(DatasetWrapper)
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


class AudioTransformer(DatasetWrapper[Any]):
    """
    Wrapper class for applying image transformations.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(self, dataset: Dataset, *, transform: Optional[AudioTform]) -> None:
        self.dataset = dataset
        self._transform: Optional[ImageTform] = None
        self.transform = transform

    @implements(DatasetWrapper)
    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if self.transform is not None:
            if isinstance(sample, SampleBase):
                if not isinstance(sample.x, Tensor):
                    raise TypeError(
                        f"Audio transform cannot be applied to input of type '{type(sample.x)}'"
                        "(must be a PyTorch Tensor)."
                    )
                waveform = apply_audio_transform(waveform=sample.x, transform=self.transform)
                sample = replace(sample, x=waveform)
            elif isinstance(sample, Tensor):
                sample = apply_audio_transform(waveform=sample, transform=self.transform)
            else:
                waveform = apply_audio_transform(waveform=sample[0], transform=self.transform)
                data_type = type(sample)
                sample = data_type(waveform, *sample[1:])
        return sample


class TabularTransformer(DatasetWrapper[Any]):
    """
    Wrapper class for applying transformations to tabular data.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Optional[TabularTransform],
        target_transform: Optional[TabularTransform],
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    @implements(DatasetWrapper)
    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if self.transform is not None:
            if isinstance(sample, SampleBase):
                if not isinstance(sample.x, Tensor):
                    raise TypeError(
                        f"Tabular transform cannot be applied to input of type '{type(sample.x)}'"
                        "(must be a PyTorch Tensor)."
                    )
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


class InstanceWeightedDataset(DatasetWrapper[Any]):
    """Wrapper endowing datasets with instance-weights."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.iw = compute_instance_weights(dataset)

    @implements(DatasetWrapper)
    def __getitem__(self, index: int) -> Union[NamedSample, Tuple[Any, ...]]:
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
