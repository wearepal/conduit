from __future__ import annotations
import random
from typing import List, cast

from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from bolts.models.self_supervised.moco import GaussianBlur

__all__ = [
    "MultiCropTransform",
    "Solarization",
]


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


class MultiCropTransform:
    def __init__(
        self,
        *,
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> None:
        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                T.RandomGrayscale(p=0.2),
            ]
        )
        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = T.Compose(
            [
                T.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = T.Compose(
            [
                T.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = T.Compose(
            [
                T.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image: Image.Image) -> list[Tensor]:
        global_crops = cast(
            List[Tensor], [self.global_transfo1(image), self.global_transfo2(image)]
        )
        local_crops: list[Tensor] = []
        for _ in range(self.local_crops_number):
            local_crops.append(cast(Tensor, self.local_transfo(image)))
        return global_crops + local_crops
