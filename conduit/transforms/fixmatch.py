from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar, Union

from PIL import Image
import numpy as np
from ranzen.misc import gcopy
from torch import Tensor
from typing_extensions import Self, override

from conduit.data.datasets.vision.utils import ImageTform, apply_image_transform
from conduit.data.structures import InputContainer, RawImage, concatenate_inputs

__all__ = [
    "FixMatchPair",
    "FixMatchTransform",
]

X = TypeVar("X", bound=Union[Tensor, RawImage, List[Image.Image]])


@dataclass
class FixMatchPair(InputContainer[X]):
    strong: X
    weak: X

    @override
    def __len__(self) -> int:
        if isinstance(self.strong, Image.Image):
            return 1
        return len(self.strong)

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        is_batched = isinstance(self.strong, (Tensor, np.ndarray)) and (self.strong.ndim == 4)
        copy.strong = concatenate_inputs(copy.strong, other.strong, is_batched=is_batched)
        copy.weak = concatenate_inputs(copy.weak, other.weak, is_batched=is_batched)

        return copy


A = TypeVar("A", bound=ImageTform)


class FixMatchTransform(Generic[A]):
    def __init__(
        self,
        strong_transform: A,
        *,
        weak_transform: A,
        shared_transform_start: Optional[Callable[[Union[RawImage, Tensor]], RawImage]] = None,
        shared_transform_end: Optional[
            Callable[[Union[RawImage, Tensor]], Union[RawImage, Tensor]]
        ] = None,
    ) -> None:
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.shared_transform_start = shared_transform_start
        self.shared_transform_end = shared_transform_end

    def __call__(self, image: RawImage) -> FixMatchPair:
        if self.shared_transform_start is not None:
            image = self.shared_transform_start(image)

        strongly_augmented_image = apply_image_transform(
            image=image, transform=self.strong_transform
        )

        weakly_augmented_image = apply_image_transform(image=image, transform=self.weak_transform)
        if self.shared_transform_end is not None:
            strongly_augmented_image = self.shared_transform_end(strongly_augmented_image)
            weakly_augmented_image = self.shared_transform_end(weakly_augmented_image)

        return FixMatchPair(strong=strongly_augmented_image, weak=weakly_augmented_image)
