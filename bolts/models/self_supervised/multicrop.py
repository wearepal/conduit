from __future__ import annotations
from dataclasses import dataclass, field

from torch import Tensor

from bolts.data.datasets.utils import (
    ImageTform,
    RawImage,
    apply_image_transform,
    img_to_tensor,
)

__all__ = ["MultiCropOutput", "MultiCropTransform"]


@dataclass(frozen=True)
class MultiCropOutput:
    global_crops: list[Tensor]
    local_crops: list[Tensor] = field(default_factory=list)

    @property
    def all_crops(self) -> list[Tensor]:
        return self.global_crops + self.local_crops

    def __len__(self) -> int:
        return len(self.global_crops) + len(self.local_crops)


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
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> MultiCropTransform:
        from bolts.models.self_supervised.dino.transforms import dino_train_transform

        return dino_train_transform(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
        )

    @classmethod
    def with_mocov2_transform(cls: type[MultiCropTransform]):
        from bolts.models.self_supervised.moco.transforms import mocov2_train_transform

        return cls(global_transform_1=mocov2_train_transform())

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
