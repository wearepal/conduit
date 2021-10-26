from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from conduit.types import NDArrayR

__all__ = [
    "Denormalize",
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
