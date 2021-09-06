from __future__ import annotations
import random
from typing import Sequence

from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from conduit.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from conduit.data.structures import MeanStd
from conduit.models.self_supervised.moco import GaussianBlur
from conduit.models.self_supervised.multicrop import MultiCropTransform

__all__ = ["Solarization", "dino_train_transform"]


class Solarization:
    """
    Apply Solarization to a PIL image with some probability.
    """

    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def dino_train_transform(
    *,
    global_crop_size: int | Sequence[int] = 224,
    local_crop_size: int | Sequence[int] = 96,
    norm_values: MeanStd | None = IMAGENET_STATS,
    global_crops_scale: tuple[float, float] = (0.4, 1.0),
    local_crops_scale: tuple[float, float] = (0.05, 0.4),
    local_crops_number: int = 8,
) -> MultiCropTransform:
    flip_and_color_jitter = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ]
    )
    normalize_ls: list[PillowTform] = [T.ToTensor()]
    if norm_values is not None:
        normalize_ls.append(
            T.Normalize(mean=norm_values.mean, std=norm_values.std),
        )
    normalize = T.Compose(normalize_ls)

    # first global crop
    global_transform_1 = T.Compose(
        [
            T.RandomResizedCrop(
                global_crop_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC
            ),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ]
    )
    # second global crop
    global_transform_2 = T.Compose(
        [
            T.RandomResizedCrop(
                local_crop_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC
            ),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ]
    )
    # transformation for the local small crops
    local_transform = T.Compose(
        [
            T.RandomResizedCrop(
                local_crop_size, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC
            ),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ]
    )
    return MultiCropTransform(
        global_transform_1=global_transform_1,
        global_transform_2=global_transform_2,
        local_transform=local_transform,
        local_crops_number=local_crops_number,
    )
