from typing import List, Optional, Sequence, Union

import numpy as np
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from conduit.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from conduit.data.structures import MeanStd
from conduit.transforms import RandomGaussianBlur

__all__ = [
    "moco_ft_transform",
    "moco_test_transform",
    "mocov2_train_transform",
]


def mocov2_train_transform(
    crop_size: Union[int, Sequence[int]] = 224, norm_values: Optional[MeanStd] = IMAGENET_STATS
) -> T.Compose:
    transform_ls: List[PillowTform] = [
        T.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
        T.RandomGrayscale(p=0.2),
        T.RandomApply([RandomGaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0)], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    if norm_values is not None:
        transform_ls.append(T.Normalize(mean=norm_values.mean, std=norm_values.std))
    return T.Compose(transform_ls)


def moco_ft_transform(
    crop_size: Union[int, Sequence[int]] = 224, norm_values: Optional[MeanStd] = IMAGENET_STATS
) -> T.Compose:
    transform_ls: List[PillowTform] = [
        T.RandomResizedCrop(crop_size, interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    if norm_values is not None:
        transform_ls.append(T.Normalize(mean=norm_values.mean, std=norm_values.std))
    return T.Compose(transform_ls)


def moco_test_transform(
    crop_size: Union[int, Sequence[int]] = 224,
    amount_to_crop: int = 32,
    norm_values: Optional[MeanStd] = IMAGENET_STATS,
) -> T.Compose:
    resized_size = np.add(crop_size, amount_to_crop).astype(np.int64)
    if isinstance(resized_size, np.integer):
        resized_size = resized_size.item()
    else:
        resized_size = tuple(resized_size)
    transform_ls: List[PillowTform] = [
        T.Resize(resized_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ]
    if norm_values is not None:
        transform_ls.append(T.Normalize(mean=norm_values.mean, std=norm_values.std))
    return T.Compose(transform_ls)
