import random
from typing import Sequence, Tuple, Union

from PIL import Image, ImageFilter, ImageOps
import torch
from torch import Tensor
import torchvision.transforms as T  # type: ignore
import torchvision.transforms.functional as TF  # type: ignore

from conduit.types import NDArrayR

__all__ = [
    "Denormalize",
    "RandomGaussianBlur",
    "RandomSolarize",
    "denormalize",
]


def _invert_norm_values(
    mean: Union[float, Tensor, Sequence[float], NDArrayR],
    std: Union[float, Tensor, Sequence[float], NDArrayR],
) -> Tuple[Tensor, Tensor]:
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    std_inv = std.reciprocal().clip(torch.finfo(std.dtype).eps)
    mean_inv = -mean * std_inv
    return mean_inv, std_inv


def denormalize(
    tensor: Tensor,
    mean: Union[float, Tensor, Sequence[float], NDArrayR],
    std: Union[float, Tensor, Sequence[float], NDArrayR],
    inplace: bool = False,
) -> Tensor:
    """Denormalize a float tensor image with mean and standard deviation.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will dennormalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] * std[channel]) + mean[channel])``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    """
    mean_inv, std_inv = _invert_norm_values(mean=mean, std=std)
    return TF.normalize(
        tensor=tensor, mean=mean_inv.tolist(), std=std_inv.tolist(), inplace=inplace
    )


class Denormalize(T.Normalize):
    """Denormalize a float tensor image with mean and standard deviation.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will dennormalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] * std[channel]) + mean[channel])``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    """

    def __init__(
        self,
        mean: Union[float, Tensor, Sequence[float], NDArrayR],
        std: Union[float, Tensor, Sequence[float], NDArrayR],
        inplace: bool = False,
    ) -> None:
        mean_inv, std_inv = _invert_norm_values(mean=mean, std=std)
        super().__init__(mean=mean_inv, std=std_inv, inplace=inplace)


class RandomGaussianBlur:
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


class RandomSolarize:
    """
    Apply Solarization to a PIL image with some probability.
    """

    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.solarize(img) if random.random() < self.p else img
