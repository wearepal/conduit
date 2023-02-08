from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar, Union, overload

from ranzen.misc import gcopy
import torch
from torch import Tensor
import torchvision.transforms as T  # type: ignore
import torchvision.transforms.functional as TF  # type: ignore
from typing_extensions import Self, override

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.vision.utils import (
    ImageTform,
    PillowTform,
    apply_image_transform,
    img_to_tensor,
)
from conduit.data.structures import (
    InputContainer,
    MeanStd,
    RawImage,
    concatenate_inputs,
)

from .image import RandomGaussianBlur, RandomSolarize

__all__ = [
    "MultiCropOutput",
    "MultiCropTransform",
    "MultiViewPair",
]


@dataclass
class MultiViewPair(InputContainer[Tensor]):
    v1: Tensor
    v2: Tensor

    def __post_init__(self) -> None:
        if self.v1.size() != self.v2.size():
            raise AttributeError("'v1' and 'v2' must have the same shape.")

    @override
    def __len__(self) -> int:
        return len(self.v1)

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        is_batched = self.v1.ndim == 4
        copy.v1 = concatenate_inputs(copy.v1, other.v1, is_batched=is_batched)
        copy.v2 = concatenate_inputs(copy.v2, other.v2, is_batched=is_batched)

        return copy

    def size(self) -> torch.Size:
        return self.v1.size()

    def shape(self) -> torch.Size:
        return self.v1.shape

    def merge(self) -> Tensor:
        is_batched = self.v1.ndim == 4
        return concatenate_inputs(self.v1, self.v2, is_batched=is_batched)

    @property
    def anchor(self) -> Tensor:
        return self.v1

    @property
    def target(self) -> Tensor:
        return self.v2

    @property
    def num_sources(self) -> int:
        return len(self)


@dataclass
class MultiCropOutput(InputContainer[MultiViewPair]):
    global_views: MultiViewPair
    local_views: Tensor

    @property
    def num_sources(self) -> int:
        """The number of samples from which the views were generated."""
        return len(self.global_views)

    @property
    def num_global_crops(self) -> int:
        return 2

    @property
    def num_local_crops(self) -> int:
        if self.local_views is None:
            return 0
        return len(self.local_views) // len(self.global_views)

    @property
    def num_crops(self) -> int:
        return self.num_global_crops + self.num_local_crops

    @property
    def global_crop_size(self) -> Tuple[int, int, int]:
        return self.global_views.shape[1:]  # type: ignore

    @property
    def local_crop_size(self) -> Tuple[int, int, int]:
        if self.local_views is None:
            raise AttributeError("Cannot retrieve the local-crop size as 'local_' is 'None'.")
        return self.local_views.shape[1:]

    @property
    def shape(self):
        """Shape of the global crops."""
        return self.global_views.shape

    def astuple(self) -> Tuple[Tensor, Tensor]:
        return (self.global_views.merge(), self.local_views)

    @property
    def anchor(self) -> Tuple[Tensor, Tensor]:
        return (self.global_views.v1, self.local_views)

    @property
    def target(self) -> Tensor:
        return self.global_views.v2

    @override
    def __len__(self) -> int:
        """Total number of crops."""
        return len(self.global_views) + len(self.local_views)

    @override
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.global_views = copy.global_views + other.global_views
        if copy.local_views is None:
            if other.local_views is not None:
                copy.local_views = other.local_views
        else:
            if other.local_views is not None:
                copy.local_views = copy.local_views + other.local_views
                is_batched = copy.local_views.ndim == 4
                copy.local_views = concatenate_inputs(
                    copy.local_views, other.local_views, is_batched=is_batched
                )
        return copy


LT = TypeVar("LT", bound=Optional[ImageTform])


class MultiCropTransform(Generic[LT]):
    def __init__(
        self,
        *,
        global_transform_1: ImageTform,
        global_transform_2: Optional[ImageTform] = None,
        local_transform: LT = None,
        local_crops_number: int = 8,
    ) -> None:
        self.global_transform_1 = global_transform_1
        self.global_transform_2 = (
            global_transform_1 if global_transform_2 is None else global_transform_2
        )
        if (local_transform is not None) and (local_crops_number <= 0):
            raise AttributeError(
                " 'local_crops' must be a positive integer if 'local_transform' is defined."
            )
        self.local_transform = local_transform
        self.local_crops_number = local_crops_number

    @staticmethod
    def _apply_transform(image: RawImage, transform: ImageTform):
        view = apply_image_transform(image, transform=transform)
        if not isinstance(view, Tensor):
            view = img_to_tensor(view)
        return view

    @overload
    def __call__(self: "MultiCropTransform[ImageTform]", image: RawImage) -> MultiCropOutput:
        ...

    @overload
    def __call__(self: "MultiCropTransform[None]", image: RawImage) -> MultiViewPair:
        ...

    def __call__(
        self: "MultiCropTransform", image: RawImage
    ) -> Union[MultiCropOutput, MultiViewPair]:
        global_crop_v1 = self._apply_transform(image, transform=self.global_transform_1)
        global_crop_v2 = self._apply_transform(image, transform=self.global_transform_2)
        gc_pair = MultiViewPair(v1=global_crop_v1, v2=global_crop_v2)

        if (self.local_transform is None) or (self.local_crops_number == 0):
            return gc_pair
        local_crops = torch.stack(
            [
                self._apply_transform(image, transform=self.local_transform)
                for _ in range(self.local_crops_number)
            ],
            dim=0,
        )

        return MultiCropOutput(global_views=gc_pair, local_views=local_crops)

    @classmethod
    def with_dino_transform(
        cls,
        *,
        global_crop_size: Union[int, Sequence[int]] = 224,
        local_crop_size: Union[int, Sequence[int]] = 96,
        norm_values: Optional[MeanStd] = IMAGENET_STATS,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> "MultiCropTransform":
        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                T.RandomGrayscale(p=0.2),
            ]
        )
        normalize_ls: List[PillowTform] = [T.ToTensor()]
        if norm_values is not None:
            normalize_ls.append(
                T.Normalize(mean=norm_values.mean, std=norm_values.std),
            )
        normalize = T.Compose(normalize_ls)

        # first global crop
        global_transform_1 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=TF.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        global_transform_2 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=TF.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(0.1),
                RandomSolarize(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        local_transform = None
        if local_crops_number > 0:
            local_transform = T.Compose(
                [
                    T.RandomResizedCrop(
                        local_crop_size,
                        scale=local_crops_scale,
                        interpolation=TF.InterpolationMode.BICUBIC,
                    ),
                    flip_and_color_jitter,
                    RandomGaussianBlur(p=0.5),
                    normalize,
                ]
            )

        return MultiCropTransform(
            global_transform_1=global_transform_1,
            global_transform_2=global_transform_2,
            local_transform=local_transform,
            local_crops_number=local_crops_number,
        )
