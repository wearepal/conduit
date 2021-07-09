from __future__ import annotations
from functools import lru_cache
import math
from pathlib import Path
from typing import Callable, Sequence, Union, overload

from PIL import Image
import albumentations as A
import cv2
from kit.torch.data import StratifiedSampler
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.transforms import functional as F
from typing_extensions import Literal, get_args

__all__ = [
    "AlbumentationsTform",
    "ImageLoadingBackend",
    "ImageTform",
    "PillowTform",
    "RawImage",
    "SizedStratifiedSampler",
    "apply_image_transform",
    "extract_base_dataset",
    "extract_labels_from_dataset",
    "get_group_ids",
    "img_to_tensor",
    "infer_il_backend",
    "load_image",
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
        if image is None:
            raise OSError(f"Image-file could not be read from location '{filepath}'")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
    return Image.open(filepath)


AlbumentationsTform = Union[A.Compose, A.BasicTransform]
PillowTform = Callable[[Image.Image], Union[Tensor, Image.Image]]
ImageTform = Union[AlbumentationsTform, PillowTform]


def infer_il_backend(transform: ImageTform | None) -> ImageLoadingBackend:
    """Infer which image-loading backend to use based on the type of the image-transform."""
    # Default to openccv is transform is None as numpy arrays are generally
    # more tractable
    if transform is None or isinstance(transform, get_args(AlbumentationsTform)):
        return "opencv"
    return "pillow"


def apply_image_transform(
    image: RawImage, transform: ImageTform | None
) -> np.ndarray | Image.Image | Tensor:
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


def img_to_tensor(img: Image.Image | np.ndarray) -> Tensor:
    if isinstance(img, Image.Image):
        return F.pil_to_tensor(img)
    return torch.from_numpy(
        np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    )


def extract_base_dataset(dataset: Dataset) -> tuple[Dataset, Tensor | slice]:
    def _closure(
        dataset: Dataset, rel_indices_ls: list[list[int]] | None = None
    ) -> tuple[Dataset, Tensor | slice]:
        if rel_indices_ls is None:
            rel_indices_ls = []
        if hasattr(dataset, "dataset"):
            if isinstance(dataset, Subset):
                rel_indices_ls.append(list(dataset.indices))
            return _closure(dataset.dataset, rel_indices_ls)  # type: ignore
        if rel_indices_ls:
            abs_indices = torch.as_tensor(rel_indices_ls.pop(), dtype=torch.long)
            for indices in rel_indices_ls[::-1]:
                abs_indices = abs_indices[indices]
        else:
            abs_indices = slice(None)
        return dataset, abs_indices

    return _closure(dataset)


@lru_cache(typed=True)
def extract_labels_from_dataset(dataset: Dataset) -> tuple[Tensor | None, Tensor | None]:
    """Attempt to extract s/y labels from a dataset."""

    def _closure(dataset: Dataset) -> tuple[Tensor | None, Tensor | None]:
        dataset, indices = extract_base_dataset(dataset)
        _s = None
        _y = None
        if getattr(dataset, "s", None) is not None:
            _s = dataset.s[indices]  # type: ignore
        if getattr(dataset, "y", None) is not None:
            _s = dataset.s[indices]  # type: ignore

        _s = torch.from_numpy(_s) if isinstance(_s, np.ndarray) else _s
        _y = torch.from_numpy(_y) if isinstance(_y, np.ndarray) else _y

        return _s, _y

    if isinstance(dataset, (ConcatDataset)):
        s_all_ls, y_all_ls = [], []
        for _dataset in dataset.datasets:
            s, y = _closure(_dataset)
            if s is not None:
                s_all_ls.append(s)
            if y is not None:
                s_all_ls.append(y)
        s_all = torch.cat(s_all_ls, dim=0) if s_all_ls else None
        y_all = torch.cat(y_all_ls, dim=0) if y_all_ls else None
    else:
        s_all, y_all = _closure(dataset)
    return s_all, y_all


def get_group_ids(dataset: Dataset) -> Tensor:
    s_all, y_all = extract_labels_from_dataset(dataset)
    group_ids = None
    if s_all is None:
        if y_all is None:
            raise ValueError(
                "Unable to compute group ids for dataset because no labels could be extracted."
            )
        group_ids = y_all
    else:
        if group_ids is None:
            group_ids = s_all
        else:
            group_ids = (group_ids * len(s_all.unique()) + s_all).squeeze()
    return group_ids


def compute_instance_weights(dataset: Dataset) -> Tensor:
    group_ids = get_group_ids(dataset)
    _, counts = group_ids.unique(return_counts=True)
    return group_ids / counts


class SizedStratifiedSampler(StratifiedSampler):
    """StratifiedSampler with a finite length for epoch-based training."""

    def __init__(
        self,
        group_ids: Sequence[int],
        num_samples_per_group: int,
        shuffle: bool = False,
        multipliers: dict[int, int] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(
            group_ids=group_ids,
            num_samples_per_group=num_samples_per_group,
            base_sampler="sequential",
            shuffle=shuffle,
            replacement=False,
            multipliers=multipliers,
            generator=generator,
        )
        # We define the legnth of the sampler to be the maximum number of steps
        # needed to do a complete pass of a group's data
        groupwise_epoch_len = (
            math.ceil(len(idxs) / (mult * num_samples_per_group))
            for idxs, mult in self.groupwise_idxs
        )
        self._max_epoch_len = max(groupwise_epoch_len)

    def __len__(self) -> int:
        return self._max_epoch_len
