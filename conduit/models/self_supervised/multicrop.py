from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

from kit.decorators import implements
import torch
from torch import Tensor
import torch.nn as nn

from conduit.constants import IMAGENET_STATS
from conduit.data.datasets.utils import (
    ImageTform,
    RawImage,
    apply_image_transform,
    img_to_tensor,
)
from conduit.data.structures import MeanStd, MultiCropOutput

__all__ = [
    "MultiCropTransform",
    "MultiCropWrapper",
]


class MultiCropTransform:
    def __init__(
        self,
        *,
        global_transform_1: ImageTform,
        global_transform_2: ImageTform | None = None,
        local_transform: ImageTform | None = None,
        local_crops_number: int = 8,
    ) -> None:
        self.global_transform_1 = global_transform_1
        self.global_transform_2 = (
            global_transform_1 if global_transform_2 is None else global_transform_2
        )
        self.local_transform = local_transform
        if local_crops_number < 0:
            raise AttributeError("'local_crops' must be a non-negative integer.")
        self.local_crops_number = local_crops_number

    @classmethod
    def with_dino_transform(
        cls: type[MultiCropTransform],
        *,
        global_crop_size: int | Sequence[int] = 224,
        local_crop_size: int | Sequence[int] = 96,
        norm_values: MeanStd | None = IMAGENET_STATS,
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> MultiCropTransform:
        from conduit.models.self_supervised.dino.transforms import dino_train_transform

        return dino_train_transform(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            norm_values=norm_values,
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
        )

    @classmethod
    def with_mocov2_transform(
        cls: type[MultiCropTransform],
        crop_size: int | Sequence[int] = 224,
        norm_values: MeanStd | None = IMAGENET_STATS,
    ) -> MultiCropTransform:
        from conduit.models.self_supervised.moco.transforms import (
            mocov2_train_transform,
        )

        return cls(
            global_transform_1=mocov2_train_transform(crop_size=crop_size, norm_values=norm_values)
        )

    def _apply_transform(self, image: RawImage, transform: ImageTform):
        view = apply_image_transform(image, transform=transform)
        if not isinstance(view, Tensor):
            view = img_to_tensor(view)
        return view

    def __call__(self, image: RawImage) -> MultiCropOutput:
        global_crops = [
            self._apply_transform(image, transform=transform)
            for transform in (self.global_transform_1, self.global_transform_2)
        ]
        local_crops: list[Tensor] = []
        if self.local_transform is not None:
            for _ in range(self.local_crops_number):
                local_crops.append(self._apply_transform(image, transform=self.local_transform))
        return MultiCropOutput(global_crops=global_crops, local_crops=local_crops)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are concatenated and a
    single forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone: nn.Module, *, head: nn.Module | None) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Identity() if head is None else head

    @implements(nn.Module)
    def forward(self, x: list[Tensor] | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            return self.head(self.backbone(x))

        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output: Tensor = _out if start_idx == 0 else torch.cat((output, _out))  # type: ignore
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)  # type: ignore
