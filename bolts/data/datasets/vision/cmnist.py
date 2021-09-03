"""ColoredMNIST Dataset."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union, overload

from PIL import Image
from ethicml.vision import LdColorizer
from kit.decorators import implements, parsable
import numpy as np
import numpy.typing as npt
import torch
from torch.functional import Tensor
from torchvision.datasets import MNIST
from typing_extensions import Literal

from bolts.data.datasets.utils import ImageTform, RawImage

from .base import PBVisionDataset

__all__ = ["ColoredMNIST"]


@overload
def _filter_data_by_labels(
    data: Tensor, *, targets: Tensor, label_map: dict[str, int], inplace: Literal[True] = ...
) -> None:
    ...


@overload
def _filter_data_by_labels(
    data: Tensor,
    *,
    targets: Tensor,
    label_map: dict[str, int],
    inplace: Literal[False] = ...,
) -> tuple[Tensor, Tensor]:
    ...


def _filter_data_by_labels(
    data: Tensor, *, targets: Tensor, label_map: dict[str, int], inplace: bool = True
) -> tuple[Tensor, Tensor] | None:
    if not inplace:
        data, targets = data.clone(), targets.clone()
    final_mask = torch.zeros_like(targets).bool()
    for old_label, new_label in label_map.items():
        mask = targets == int(old_label)
        targets[mask] = new_label
        final_mask |= mask
    data = data[final_mask]
    targets = targets[final_mask]
    if inplace:
        return data, targets
    return None


class ColoredMNIST(PBVisionDataset):
    x: npt.NDArray[np.floating]

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        label_map: Optional[Dict[str, int]] = None,
        colors: Optional[List[int]] = None,
        num_colors: int = 10,
        scale: float = 0.2,
        train: bool = True,
        correlation: float = 1.0,
        binarize: bool = False,
        greyscale: bool = False,
        background: bool = False,
        black: bool = True,
    ) -> None:
        base_dataset = MNIST(root=str(root), download=download, train=train)
        self.label_map = label_map
        self.scale = scale
        self.num_colors = num_colors
        self.colors = colors
        self.background = background
        self.binarize = binarize
        self.black = black
        self.greyscale = greyscale
        if not 0 <= correlation <= 1:
            raise ValueError(
                "Strength of correlation between colour and targets must be between 0 and 1."
            )
        self.correlation = correlation

        # Convert the greyscale iamges of shape ( H, W ) into 'colour' images of shape ( C, H, W )
        x_tiled = base_dataset.data
        y = base_dataset.targets
        if self.label_map is not None:
            _filter_data_by_labels(data=x_tiled, targets=y, label_map=self.label_map, inplace=True)
        s = y % self.num_colors

        if self.correlation < 1:
            # Change the values of randomly-selected labeld to values other than their original ones
            to_flip = torch.rand(s.size(0)) > self.correlation
            s[to_flip] += (
                torch.randint(  # type: ignore
                    low=1, high=self.num_colors, size=(to_flip.count_nonzero(),)
                )
                % self.num_colors
            )

        # Colorize the greyscale images
        colorizer = LdColorizer(
            scale=self.scale,
            background=self.background,
            black=self.black,
            binarize=self.binarize,
            greyscale=self.greyscale,
            color_indices=self.colors,
        )

        x_tiled = x_tiled.unsqueeze(1).expand(-1, 3, -1, -1)
        x_colorized = colorizer(data=x_tiled, labels=s)
        x_colorized = x_colorized.movedim(1, -1).numpy().astype(np.uint8)

        super().__init__(x=x_colorized, y=y, s=s, transform=transform, image_dir=root)

    @implements(PBVisionDataset)
    def _load_image(self, index: int) -> RawImage:
        image = self.x[index]
        if self._il_backend == "pillow":
            image = Image.fromarray(image)
        return image
