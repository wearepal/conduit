from __future__ import annotations
import random
from typing import Sequence

from PIL import Image, ImageFilter
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from bolts.data import ImageTform, apply_image_transform, img_to_tensor

__all__ = [
    "GaussianBlur",
    "TwoCropsTransform",
    "moco_eval_transform",
    "mocov2_train_transform",
]


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


def mocov2_train_transform() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0)], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TwoCropsTransform:
    def __init__(self, transform: ImageTform) -> None:
        self.transform = transform

    @classmethod
    def with_mocov2_transform(cls: type[TwoCropsTransform]):
        return cls(transform=mocov2_train_transform())

    def __call__(self, anchor: Image.Image) -> list[Tensor]:
        views: list[Tensor] = []
        for _ in range(2):
            view = apply_image_transform(image=anchor, transform=self.transform)
            if not isinstance(view, Tensor):
                if isinstance(view, Sequence):
                    view = [img_to_tensor(subview) for subview in view]
                else:
                    view = img_to_tensor(view)
            if isinstance(view, Sequence):
                views.extend(view)
            else:
                views.append(view)
        return views


def moco_eval_transform(train: bool) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    return T.Compose(
        [
            T.Resize(256, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
