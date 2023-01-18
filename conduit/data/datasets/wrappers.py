"""Dataset wrappers."""
from dataclasses import is_dataclass, replace
from typing import Any, Optional, Tuple, Union

from torch import Tensor
from typing_extensions import override

from conduit.data.datasets.utils import (
    AudioTform,
    apply_audio_transform,
    compute_instance_weights,
)
from conduit.data.structures import (
    BinarySample,
    BinarySampleIW,
    Dataset,
    DatasetWrapper,
    NamedSample,
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
    "InstanceWeightedDataset",
    "TabularTransformer",
]


class AudioTransformer(DatasetWrapper[Any]):
    """
    Wrapper class for applying image transformations.

    Useful when wanting to have different transformations for different subsets of the data
    that share the same underlying dataset.
    """

    def __init__(self, dataset: Dataset, *, transform: Optional[AudioTform]) -> None:
        self.dataset = dataset
        self._transform: Optional[AudioTform] = None
        self.transform = transform

    @override
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

    @override
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

    @override
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
