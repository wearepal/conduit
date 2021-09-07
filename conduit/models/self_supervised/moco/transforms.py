from __future__ import annotations
import random
from typing import Sequence

from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from conduit.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from conduit.data.structures import MeanStd

__all__ = ["moco_ft_transform", "moco_test_transform", "mocov2_train_transform", "GaussianBlur"]


class GaussianBlur:
    """
    Apply Gaussian Blur to the PIL image with some probability.
    """

    def __init__(self, p: float = 0.5, *, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


def mocov2_train_transform(
    crop_size: int | Sequence[int] = 224, norm_values: MeanStd | None = IMAGENET_STATS
) -> T.Compose:
    transform_ls: list[PillowTform] = [
        T.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0)], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    if norm_values is not None:
        transform_ls.append(T.Normalize(mean=norm_values.mean, std=norm_values.std))
    return T.Compose(transform_ls)


def moco_ft_transform(
    crop_size: int | Sequence[int] = 224, norm_values: MeanStd | None = IMAGENET_STATS
) -> T.Compose:
    transform_ls: list[PillowTform] = [
        T.RandomResizedCrop(crop_size, interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    if norm_values is not None:
        transform_ls.append(T.Normalize(mean=norm_values.mean, std=norm_values.std))
    return T.Compose(transform_ls)


def moco_test_transform(
    crop_size: int = 224,
    amount_to_crop: int = 32,
    norm_values: MeanStd | None = IMAGENET_STATS,
) -> T.Compose:
    transform_ls: list[PillowTform] = [
        T.Resize(crop_size + amount_to_crop, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ]
    if norm_values is not None:
        transform_ls.append(T.Normalize(mean=norm_values.mean, std=norm_values.std))
    return T.Compose(transform_ls)
