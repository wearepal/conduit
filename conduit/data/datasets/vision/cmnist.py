"""ColoredMNIST Dataset."""
from enum import auto
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union, cast

from PIL import Image
import numpy as np
import numpy.typing as npt
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from torchvision.datasets import MNIST  # type: ignore
from typing_extensions import TypeAlias, override

from conduit.data.structures import TernarySample
from conduit.types import NDArrayR

from .base import CdtVisionDataset
from .utils import ImageTform, RawImage

__all__ = ["ColoredMNIST", "ColoredMNISTSplit", "MNISTColorizer"]


class MNISTColorizer:
    """Convert a greyscale MNIST image to RGB."""

    COLORS: ClassVar[Tensor] = torch.tensor(
        [
            (0, 255, 255),
            (0, 0, 255),  # blue
            (255, 0, 255),
            (0, 128, 0),
            (0, 255, 0),  # green
            (128, 0, 0),
            (0, 0, 128),
            (128, 0, 128),
            (255, 0, 0),  # red
            (255, 255, 0),  # yellow
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        scale: float,
        *,
        min_val: float = 0.0,
        max_val: float = 1.0,
        binarize: bool = False,
        background: bool = False,
        black: bool = True,
        greyscale: bool = False,
        color_indices: Optional[Union[List[int], slice]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Colorizes a grayscale image by sampling colors from multivariate normal distributions.

        The distribution is centered on predefined means and standard deviation determined by the
        scale argument.

        :param min_val: Minimum value the input data can take (needed for clamping).
        :param max_val: Maximum value the input data can take (needed for clamping).
        :param scale: Standard deviation of the multivariate normal distributions from which
            the colors are drawn. Lower values correspond to higher bias. Defaults to 0.02.
        :param binarize: Whether to the binarize the grayscale data before colorisation.
        :param background: Whether to color the background instead of the foreground.
        :param black: Whether not to invert the black. Defaults to True.
        :param greyscale: Whether to greyscale the colorised images. Defaults to False.
        :param color_indices: Choose specific colors if you don't need all 10.
        :param seed: Random seed used for sampling colors.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.binarize = binarize
        self.background = background
        self.black = black
        self.greyscale = greyscale
        self.scale = scale
        self.seed = seed

        self.generator = (
            torch.default_generator
            if self.seed is None
            else torch.Generator().manual_seed(self.seed)
        )
        self.palette = self.COLORS if color_indices is None else self.COLORS[color_indices]

    def _sample_colors(self, mean_color_values: Tensor) -> Tensor:
        return (
            torch.normal(mean=mean_color_values, std=self.scale, generator=self.generator).clip(
                0, 255
            )
            / 255.0
        )

    def __call__(
        self, images: Union[Tensor, NDArrayR], *, labels: Union[Tensor, NDArrayR]
    ) -> Tensor:
        """Apply the transformation.

        :param images:  Greyscale images to be colorized. Expected to be unnormalized (in the range
            [0, 255]).
        :param labels: Indexes (0-9) indicating the gaussian distribution from which to sample each
            image's color.
        :returns: Images converted to RGB.
        """
        if isinstance(images, np.ndarray):
            images = torch.as_tensor(images, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.as_tensor(labels, dtype=torch.long)
        # Add a singular channel dimension if one isn't already there.
        images = cast(Tensor, torch.atleast_3d(images))
        if images.ndim == 3:
            images = images.unsqueeze(1)
        images = images.expand(-1, 3, -1, -1)

        colors = self._sample_colors(self.palette[labels]).view(-1, 3, 1, 1)

        if self.binarize:
            images = (images > 127).float()

        if self.background:
            if self.black:
                # colorful background, black digits
                images_colorized = (1 - images) * colors
            else:
                # colorful background, white digits
                images_colorized = images + colors
        elif self.black:
            # black background, colorful digits
            images_colorized = images * colors
        else:
            # white background, colorful digits
            images_colorized = 1 - images * (1 - colors)

        if self.greyscale:
            images_colorized = images_colorized.mean(dim=1, keepdim=True)

        return images_colorized


def _filter_data_by_labels(
    data: Tensor,
    *,
    targets: Tensor,
    label_map: Dict[str, int],
) -> Tuple[Tensor, Tensor]:
    final_mask = torch.zeros_like(targets).bool()
    for old_label, new_label in label_map.items():
        mask = targets == int(old_label)
        targets[mask] = new_label
        final_mask |= mask
    return data[final_mask], targets[final_mask]


class ColoredMNISTSplit(StrEnum):
    TRAIN = auto()
    TEST = auto()


SampleType: TypeAlias = TernarySample


class ColoredMNIST(CdtVisionDataset[SampleType, Tensor, Tensor]):
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
        correlation: Optional[float] = None,
        binarize: bool = False,
        greyscale: bool = False,
        background: bool = False,
        black: bool = True,
        split: Optional[Union[ColoredMNISTSplit, str, List[int]]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.split = ColoredMNISTSplit(split) if isinstance(split, str) else split
        self.label_map = label_map
        self.scale = scale
        self.num_colors = num_colors
        self.colors = colors
        self.background = background
        self.binarize = binarize
        self.black = black
        self.greyscale = greyscale
        self.seed = seed
        # Note: a correlation coefficient of '1' corresponds to perfect correlation between
        # digit and class while a correlation coefficient of '-1' corresponds to perfect
        # anti-correlation.
        if correlation is None:
            correlation = 1.0 if split is ColoredMNISTSplit.TRAIN else 0.5
        if not 0 <= correlation <= 1:
            raise ValueError(
                "Strength of correlation between colour and targets must be between 0 and 1."
            )
        self.correlation = correlation

        if isinstance(self.split, ColoredMNISTSplit):
            base_dataset = MNIST(
                root=str(root), download=download, train=self.split is ColoredMNISTSplit.train
            )
            x = base_dataset.data
            y = base_dataset.targets
        else:
            x_ls, y_ls = [], []
            for _split in ColoredMNISTSplit:
                base_dataset = MNIST(
                    root=str(root), download=download, train=_split is ColoredMNISTSplit.TRAIN
                )
                x_ls.append(base_dataset.data)
                y_ls.append(base_dataset.targets)
            x = torch.cat(x_ls, dim=0)
            y = torch.cat(y_ls, dim=0)
            # custom split
            if self.split is not None:
                x = x[self.split]
                y = y[self.split]

        if self.label_map is not None:
            x, y = _filter_data_by_labels(data=x, targets=y, label_map=self.label_map)
        s = y % self.num_colors
        s_unique, s_unique_inv = s.unique(return_inverse=True)

        generator = (
            torch.default_generator
            if self.seed is None
            else torch.Generator().manual_seed(self.seed)
        )
        inv_card_s = 1 / len(s_unique)
        if self.correlation < 1:
            flip_prop = self.correlation * (1.0 - inv_card_s) + inv_card_s
            # Change the values of randomly-selected labels to values other than their original ones
            num_to_flip = round((1 - flip_prop) * len(s))
            to_flip = torch.randperm(len(s), generator=generator)[:num_to_flip]
            s_unique_inv[to_flip] += torch.randint(low=1, high=len(s_unique), size=(num_to_flip,))
            # s labels live inside the Z/(num_colors * Z) ring
            s_unique_inv[to_flip] %= len(s_unique)
            s = s_unique[s_unique_inv]

        # Convert the greyscale images of shape ( H, W ) into 'colour' images of shape ( C, H, W )
        colorizer = MNISTColorizer(
            scale=self.scale,
            background=self.background,
            black=self.black,
            binarize=self.binarize,
            greyscale=self.greyscale,
            color_indices=self.colors,
            seed=self.seed,
        )
        x_colorized = colorizer(images=x, labels=s)
        # Convert to HWC format for compatibility with transforms
        x_colorized = x_colorized.movedim(1, -1).numpy().astype(np.uint8)

        super().__init__(x=x_colorized, y=y, s=s, transform=transform, image_dir=root)

    @override
    def _load_image(self, index: int) -> RawImage:
        image = self.x[index]
        if self._il_backend == "pillow":
            image = Image.fromarray(image)
        return image
