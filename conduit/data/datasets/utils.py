from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
import logging
from multiprocessing.context import BaseContext
from pathlib import Path
import platform
from typing import Any, Callable, List, NamedTuple, Sequence, Union, cast, overload

from PIL import Image
import albumentations as A
import cv2
from kit.misc import gcopy
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
    string_classes,
)
from torch.utils.data.dataloader import DataLoader, _worker_init_fn_t
from torch.utils.data.sampler import Sampler
from torchvision.transforms import functional as TF
from typing_extensions import Literal, get_args

from conduit.data.datasets.base import CdtDataset
from conduit.data.structures import BinarySample, NamedSample, SampleBase, TernarySample

__all__ = [
    "AlbumentationsTform",
    "AudioTform",
    "CdtDataLoader",
    "FileInfo",
    "ImageLoadingBackend",
    "ImageTform",
    "PillowTform",
    "RawImage",
    "apply_image_transform",
    "check_integrity",
    "download_from_gdrive",
    "extract_base_dataset",
    "extract_labels_from_dataset",
    "get_group_ids",
    "img_to_tensor",
    "infer_al_backend",
    "infer_il_backend",
    "load_image",
    "make_subset",
    "pb_collate",
]


ImageLoadingBackend = Literal["opencv", "pillow"]


RawImage = Union[npt.NDArray[np.integer], Image.Image]


@overload
def load_image(filepath: Path | str, *, backend: Literal["opencv"] = ...) -> np.ndarray:
    ...


@overload
def load_image(filepath: Path | str, *, backend: Literal["pillow"] = ...) -> Image.Image:
    ...


def load_image(filepath: Path | str, *, backend: ImageLoadingBackend = "opencv") -> RawImage:
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
PillowTform = Callable[[Image.Image], Any]
ImageTform = Union[AlbumentationsTform, PillowTform]


def infer_il_backend(transform: ImageTform | None) -> ImageLoadingBackend:
    """Infer which image-loading backend to use based on the type of the image-transform."""
    # Default to openccv is transform is None as numpy arrays are generally
    # more tractable
    if transform is None or isinstance(transform, get_args(AlbumentationsTform)):
        return "opencv"
    return "pillow"


def apply_image_transform(
    image: RawImage, *, transform: ImageTform | None
) -> RawImage | Image.Image | Tensor:
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
        return TF.pil_to_tensor(img)
    return torch.from_numpy(
        np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    )


AudioLoadingBackend = Literal["sox_io", "soundfile"]


def infer_al_backend() -> AudioLoadingBackend:
    """Infer which audio-loading backend to use based on the operating system."""
    return 'soundfile' if platform.system() == 'Windows' else 'sox_io'


AudioTform = Callable[[Tensor], Tensor]


def apply_waveform_transform(waveform: Tensor, *, transform: AudioTform | None) -> Tensor:
    return waveform if transform is None else transform(waveform)


@overload
def extract_base_dataset(
    dataset: Dataset, *, return_subset_indices: Literal[True] = ...
) -> tuple[Dataset, Tensor | slice]:
    ...


@overload
def extract_base_dataset(
    dataset: Dataset, *, return_subset_indices: Literal[False] = ...
) -> Dataset:
    ...


def extract_base_dataset(
    dataset: Dataset, *, return_subset_indices: bool = True
) -> Dataset | tuple[Dataset, Tensor | slice]:
    def _closure(
        dataset: Dataset, rel_indices_ls: list[list[int]] | None = None
    ) -> Dataset | tuple[Dataset, Tensor | slice]:
        if rel_indices_ls is None:
            rel_indices_ls = []
        if hasattr(dataset, "dataset"):
            if isinstance(dataset, Subset):
                rel_indices_ls.append(list(dataset.indices))
            return _closure(dataset.dataset, rel_indices_ls)  # type: ignore
        if return_subset_indices:
            if rel_indices_ls:
                abs_indices = torch.as_tensor(rel_indices_ls.pop(), dtype=torch.long)
                for indices in rel_indices_ls[::-1]:
                    abs_indices = abs_indices[indices]
            else:
                abs_indices = slice(None)
            return dataset, abs_indices
        return dataset

    return _closure(dataset)


@lru_cache(typed=True)
def extract_labels_from_dataset(dataset: Dataset) -> tuple[Tensor | None, Tensor | None]:
    """Attempt to extract s/y labels from a dataset."""

    def _closure(dataset: Dataset) -> tuple[Tensor | None, Tensor | None]:
        dataset, indices = extract_base_dataset(dataset=dataset, return_subset_indices=True)
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
        s_all_ls: list[Tensor] = []
        y_all_ls: list[Tensor] = []
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
    group_ids: Tensor | None = None
    if s_all is None:
        if y_all is None:
            raise ValueError(
                "Unable to compute group ids for dataset because no labels could be extracted."
            )
        group_ids = y_all
    elif group_ids is None:
        group_ids = s_all
    else:
        group_ids = (group_ids * len(s_all.unique()) + s_all).squeeze()
    return group_ids.long()


def compute_instance_weights(dataset: Dataset, upweight: bool = False) -> Tensor:
    group_ids = get_group_ids(dataset)
    _, counts = group_ids.unique(return_counts=True)
    # Upweight samples according to the cardinality of their intersectional group
    if upweight:
        group_weights = len(group_ids) / counts
    # Downwegith samples according to the cardinality of their intersectional group
    # - this approach should be preferred due to being more numerically stable
    # (very small counts can lead to very large weighted loss values when upweighting)
    else:
        group_weights = 1 - (counts / len(group_ids))
    return group_weights[group_ids]


def make_subset(
    dataset: CdtDataset | Subset,
    *,
    indices: list[int] | npt.NDArray[np.uint64] | Tensor | slice | None,
    deep: bool = False,
) -> CdtDataset:
    if isinstance(indices, (np.ndarray, Tensor)):
        if not indices.ndim > 1:
            raise ValueError("If 'indices' is an array it must be a 0- or 1-dimensional.")
        indices = cast(List[int], indices.tolist())

    current_indices = None
    if isinstance(dataset, Subset):
        base_dataset, current_indices = extract_base_dataset(dataset, return_subset_indices=True)
        if not isinstance(base_dataset, CdtDataset):
            raise TypeError(
                f"Subsets can only be created with cdt_subset from {CdtDataset.__name__} instances "
                f"or PyTorch Subsets of them."
            )
        if isinstance(current_indices, Tensor):
            current_indices = current_indices.tolist()
    else:
        base_dataset = dataset
    subset = gcopy(base_dataset, deep=deep)

    def _subset_from_indices_(_indices: list[int] | slice) -> None:
        subset.x = subset.x[_indices]
        if subset.y is not None:
            subset.y = subset.y[_indices]
        if subset.s is not None:
            subset.s = subset.s[_indices]

    if current_indices is not None:
        _subset_from_indices_(current_indices)
    if indices is not None:
        _subset_from_indices_(indices)

    return subset


class pb_collate:
    def __init__(self, cast_to_sample: bool = True) -> None:
        self.cast_to_sample = cast_to_sample

    def _collate(self, batch: Sequence[Any]) -> Any:
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            ndims = elem.dim()
            if (ndims > 0) and ((ndims % 2) == 0):
                return torch.cat(batch, dim=0, out=out)
            return torch.stack(batch, dim=0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            elem = batch[0]
            if elem_type.__name__ == "ndarray":
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                return self._collate([torch.as_tensor(b) for b in batch])
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self._collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(**(self._collate(samples) for samples in zip(*batch)))
        elif is_dataclass(elem):  # dataclass
            return elem_type(
                **{
                    field.name: self._collate([getattr(d, field.name) for d in batch])
                    for field in fields(elem)
                }
            )
        elif isinstance(elem, (tuple, list)):
            transposed = zip(*batch)
            return [self._collate(samples) for samples in transposed]
        raise TypeError(default_collate_err_msg_format.format(elem_type))

    def __call__(self, batch: Sequence[Any]) -> Any:
        collated_batch = self._collate(batch=batch)
        if self.cast_to_sample and (not isinstance(collated_batch, SampleBase)):
            if isinstance(collated_batch, Tensor):
                collated_batch = NamedSample(x=collated_batch)
            elif isinstance(collated_batch, (tuple, list, dict)):
                if len(collated_batch) == 1:
                    sample_cls = NamedSample
                elif len(collated_batch) == 2:
                    sample_cls = BinarySample
                elif len(collated_batch) == 3:
                    sample_cls = TernarySample
                else:
                    raise ValueError
                if isinstance(collated_batch, dict):
                    collated_batch = sample_cls(**collated_batch)
                else:
                    collated_batch = sample_cls(*collated_batch)
            else:
                raise ValueError(
                    f"batch of type '{type(collated_batch)}' could not be automatically converted into a "
                    "'Sample' instance. Batch must be of type 'dict', 'tuple', or 'list'."
                )
        return collated_batch


class CdtDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int | None,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        batch_sampler: Sampler[Sequence[int]] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context: BaseContext | str | None = None,
        generator: torch.Generator | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        cast_to_sample: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=pb_collate(cast_to_sample=cast_to_sample),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


def check_integrity(*, filepath: Path, md5: str | None) -> None:
    from torchvision.datasets.utils import check_integrity  # type: ignore

    ext = filepath.suffix
    if ext not in [".zip", ".7z"] and check_integrity(fpath=str(filepath), md5=md5):
        raise RuntimeError('Dataset corrupted; try deleting it and redownloading it.')


class FileInfo(NamedTuple):
    name: str
    id: str
    md5: str | None = None


def download_from_gdrive(
    *,
    file_info: FileInfo | list[FileInfo],
    root: Path | str,
    logger: logging.Logger | None = None,
) -> None:
    """Attempt to download data if files cannot be found in the root directory."""

    logger = logging.getLogger(__name__) if logger is None else logger

    file_info_ls = file_info if isinstance(file_info, list) else [file_info]
    if not isinstance(root, Path):
        root = Path(root)
    # Create the specified root directory if it doesn't already exist
    root.mkdir(parents=True, exist_ok=True)

    for info in file_info_ls:
        filepath = root / info.name
        if not filepath.exists():
            import gdown

            logger.info(f"Downloading file '{info.name}' from Google Drive.")
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={info.id}",
                path=str(filepath),
                quiet=False,
                md5=info.md5,
            )
        else:
            logger.info(f"File '{info.name}' already downloaded.")
        if filepath.suffix == ".zip":
            if filepath.with_suffix("").exists():
                logger.info(f"File '{info.name}' already unzipped.")
            else:
                check_integrity(filepath=filepath, md5=info.md5)
                # ------------------------------ Unzip the data ------------------------------
                import zipfile

                logger.info(f"Unzipping '{filepath.resolve()}'; this could take a while.")
                with zipfile.ZipFile(filepath, "r") as fhandle:
                    fhandle.extractall(str(root))
