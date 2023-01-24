"""Utils for vision datasets."""
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, get_args, overload

from PIL import Image
import albumentations as A  # type: ignore
import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torchvision.transforms import functional as TF  # type: ignore
from typing_extensions import TypeAlias

from conduit.data.structures import RawImage

__all__ = [
    "AlbumentationsTform",
    "ImageLoadingBackend",
    "ImageTform",
    "PillowTform",
    "apply_image_transform",
    "img_to_tensor",
    "infer_il_backend",
    "load_image",
]


ImageLoadingBackend: TypeAlias = Literal["opencv", "pillow"]


@overload
def load_image(
    filepath: Union[Path, str], *, backend: Literal["opencv"] = ...
) -> npt.NDArray[np.integer]:
    ...


@overload
def load_image(filepath: Union[Path, str], *, backend: Literal["pillow"] = ...) -> Image.Image:
    ...


def load_image(filepath: Union[Path, str], *, backend: ImageLoadingBackend = "opencv") -> RawImage:
    """Load an image from disk using the requested backend.

    :param filepath: The path of the image-file to be loaded.
    :param backend: Backed to use for loading the image: either 'opencv' or 'pillow'.

    :returns: The loaded image file as a numpy array if 'opencv' was the selected backend
        and a PIL image otherwise.
    :raises OSError: if the images can't be read.
    """
    if backend == "opencv":
        if isinstance(filepath, Path):
            # cv2 can only read string filepaths
            filepath = str(filepath)
        image = cv2.imread(filepath)
        if image is None:
            raise OSError(f"Image-file could not be read from location '{filepath}'")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
    return Image.open(filepath)


AlbumentationsTform: TypeAlias = Union[A.Compose, A.BasicTransform]
PillowTform: TypeAlias = Callable[[Image.Image], Any]
ImageTform: TypeAlias = Union[AlbumentationsTform, PillowTform]


def infer_il_backend(transform: Optional[ImageTform]) -> ImageLoadingBackend:
    """Infer which image-loading backend to use based on the type of the image-transform.

    :param transform: The image transform from which to infer the image-loading backend.
        If the transform is derived from Albumentations, then 'opencv' will be selected as the
        backend, else 'pillow' will be selected.

    :returns: The backend to load images with based on the supplied image-transform: either
        'opencv' or 'pillow'.
    """
    # Default to openccv is transform is None as numpy arrays are generally
    # more tractable
    if transform is None or isinstance(transform, get_args(AlbumentationsTform)):
        return "opencv"
    return "pillow"


def apply_image_transform(
    image: RawImage, *, transform: Optional[ImageTform]
) -> Union[RawImage, Tensor]:
    image_ = image
    if transform is not None:
        if isinstance(transform, (A.Compose, A.BasicTransform)):
            if isinstance(image, Image.Image):
                image = np.array(image)
            image_ = transform(image=image)["image"]
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_ = transform(image)
    return image_


def img_to_tensor(img: Union[Image.Image, np.ndarray]) -> Tensor:
    if isinstance(img, Image.Image):
        return TF.pil_to_tensor(img)
    return torch.from_numpy(
        np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    )
