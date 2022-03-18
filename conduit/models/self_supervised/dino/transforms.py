from typing import List, Optional, Sequence, Tuple, Union

from torchvision import transforms as T  # type: ignore
from torchvision.transforms.functional import InterpolationMode  # type: ignore

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from conduit.data.structures import MeanStd
from conduit.models.self_supervised.multicrop import MultiCropTransform
from conduit.transforms import RandomGaussianBlur, RandomSolarize

__all__ = ["dino_train_transform"]


def dino_train_transform(
    *,
    global_crop_size: Union[int, Sequence[int]] = 224,
    local_crop_size: Union[int, Sequence[int]] = 96,
    norm_values: Optional[MeanStd] = IMAGENET_STATS,
    global_crops_scale: Tuple[float, float] = (0.4, 1.0),
    local_crops_scale: Tuple[float, float] = (0.05, 0.4),
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
                global_crop_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC
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
                local_crop_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC
            ),
            flip_and_color_jitter,
            RandomGaussianBlur(0.1),
            RandomSolarize(0.2),
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
