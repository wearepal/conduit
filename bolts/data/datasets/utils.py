from __future__ import annotations
from pathlib import Path
from typing import Callable, NamedTuple, Union, overload

from PIL import Image
import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
from torch import Tensor
from typing_extensions import Literal, get_args

__all__ = [
    "AlbumentationsTform",
    "ImageLoadingBackend",
    "ImageTform",
    "PillowTform",
    "infer_il_backend",
    "load_image",
    "RawImage",
    "BinarySample",
    "TernarySample",
    "apply_image_transform",
]


ImageLoadingBackend = Literal["opencv", "pillow"]


RawImage = Union[npt.NDArray[np.int_], Image.Image]


@overload
def load_image(filepath: Path | str, backend: Literal["opencv"] = ...) -> np.ndarray:
    ...


@overload
def load_image(filepath: Path | str, backend: Literal["pillow"] = ...) -> Image.Image:
    ...


def load_image(filepath: Path | str, backend: ImageLoadingBackend = "opencv") -> RawImage:
    if backend == "opencv":
        if isinstance(filepath, Path):
            # cv2 can only read string filepaths
            filepath = str(filepath)
        image = cv2.imread(filepath)  # type: ignore
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
    return Image.open(filepath)


AlbumentationsTform = Union[A.Compose, A.BasicTransform]
PillowTform = Callable[[Union[Image.Image, Tensor]], Union[Tensor, Image.Image]]
ImageTform = Union[AlbumentationsTform, PillowTform]


def infer_il_backend(transform: ImageTform | None) -> ImageLoadingBackend:
    """Infer which image-loading backend to use based on the type of the image-transform."""
    # Default to openccv is transform is None as numpy arrays are generally
    # more tractable
    if transform is None or isinstance(transform, get_args(AlbumentationsTform)):
        return "opencv"
    return "pillow"


class BinarySample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    y: Tensor | float


class TernarySample(NamedTuple):
    x: Tensor | np.ndarray | Image.Image
    s: Tensor | float
    y: Tensor | float


def apply_image_transform(
    image: RawImage, transform: ImageTform
) -> np.ndarray | Image.Image | Tensor:
    # If the image is a numpy array,  the opencv was inferred as the image-loading
    # backend and the transformation is derived from albumentations
    if transform is None:
        if isinstance(image, np.ndarray):
            image = transform(image=image)["image"]
        else:
            image = transform(image)
    return image
