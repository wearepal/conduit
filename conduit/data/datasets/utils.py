from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
import logging
from multiprocessing.context import BaseContext
from pathlib import Path
import platform
import subprocess
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)
from zipfile import BadZipFile

from PIL import Image
import albumentations as A  # type: ignore
import cv2
import numpy as np
import numpy.typing as npt
from ranzen.misc import gcopy
from ranzen.torch.data import Subset, prop_random_split
import torch
from torch import Tensor
from torch._six import string_classes
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)
from torch.utils.data.dataloader import DataLoader, _worker_init_fn_t
from torch.utils.data.sampler import Sampler
from torchvision.datasets.utils import (  # type: ignore
    _detect_file_type,
    download_url,
    extract_archive,
)
from torchvision.transforms import functional as TF  # type: ignore
from typing_extensions import TypeAlias, TypeGuard

from conduit.data.datasets.base import CdtDataset
from conduit.data.structures import (
    BinarySample,
    Dataset,
    LoadedData,
    NamedSample,
    PseudoCdtDataset,
    RawImage,
    SampleBase,
    SizedDataset,
    TernarySample,
    TrainTestSplit,
)

__all__ = [
    "AlbumentationsTform",
    "AudioTform",
    "CdtDataLoader",
    "GdriveFileInfo",
    "ImageLoadingBackend",
    "ImageTform",
    "PillowTform",
    "UrlFileInfo",
    "apply_image_transform",
    "cdt_collate",
    "check_integrity",
    "download_from_gdrive",
    "download_from_url",
    "extract_base_dataset",
    "extract_labels_from_dataset",
    "get_group_ids",
    "img_to_tensor",
    "infer_al_backend",
    "infer_il_backend",
    "infer_sample_cls",
    "is_tensor_list",
    "load_image",
    "make_subset",
    "stratified_split",
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

    :param: The path of the image-file to be loaded.
    :param backend: Backed to use for loading the image: either 'opencv' or 'pillow'.

    :returns: The loaded image file as a numpy array if 'opencv' was the selected backend
    and a PIL image otherwise.
    """
    if backend == "opencv":
        if isinstance(filepath, Path):
            # cv2 can only read string filepaths
            filepath = str(filepath)
        image = cv2.imread(filepath)  # type: ignore
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


AudioLoadingBackend: TypeAlias = Literal["sox_io", "soundfile"]


def infer_al_backend() -> AudioLoadingBackend:
    """Infer which audio-loading backend to use based on the operating system."""
    soundfile: Final = "soundfile"
    sox: Final = "sox_io"
    return soundfile if platform.system() == "Windows" else sox


AudioTform: TypeAlias = Callable[[Tensor], Tensor]


def apply_audio_transform(waveform: Tensor, *, transform: Optional[AudioTform]) -> Tensor:
    return waveform if transform is None else transform(waveform)


@overload
def extract_base_dataset(
    dataset: Dataset, *, return_subset_indices: Literal[True] = ...
) -> Tuple[Dataset, Union[Tensor, slice]]:
    ...


@overload
def extract_base_dataset(
    dataset: Dataset, *, return_subset_indices: Literal[False] = ...
) -> Dataset:
    ...


def extract_base_dataset(
    dataset: Dataset, *, return_subset_indices: bool = True
) -> Union[Dataset, Tuple[Dataset, Union[Tensor, slice]]]:
    """Extract the innermost dataset of a nesting of datasets.

    Nested datasets are inferred based on the existence of a 'dataset'
    attribute and the base dataset is extracted by recursive application
    of this rule.

    :param dataset: The dataset from which to extract the base dataset.

    :param return_subset_indices: Whether to return the indices from which
    the overall subset of the dataset was created (works for multiple levels of
    subsetting).

    :returns: The base dataset, which may be the original dataset if one does not
    exist or cannot be determined.
    """

    def _closure(
        dataset: Dataset, rel_indices_ls: Optional[List[List[int]]] = None
    ) -> Union[Dataset, Tuple[Dataset, Union[Tensor, slice]]]:
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
def extract_labels_from_dataset(
    dataset: PseudoCdtDataset,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Attempt to extract s/y labels from a dataset."""
    base_dataset = dataset

    def _closure(dataset: Dataset) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        base_dataset, indices = extract_base_dataset(dataset=dataset, return_subset_indices=True)
        _s = None
        _y = None
        if getattr(base_dataset, "s", None) is not None:
            _s = base_dataset.s[indices]  # type: ignore
        if getattr(base_dataset, "y", None) is not None:
            _y = base_dataset.y[indices]  # type: ignore

        _s = torch.from_numpy(_s) if isinstance(_s, np.ndarray) else _s
        _y = torch.from_numpy(_y) if isinstance(_y, np.ndarray) else _y

        return _s, _y

    if isinstance(base_dataset, (ConcatDataset)):
        s_all_ls: List[Tensor] = []
        y_all_ls: List[Tensor] = []
        for _dataset in base_dataset.datasets:
            s, y = _closure(_dataset)
            if s is not None:
                s_all_ls.append(s)
            if y is not None:
                s_all_ls.append(y)
        s_all = torch.cat(s_all_ls, dim=0) if s_all_ls else None
        y_all = torch.cat(y_all_ls, dim=0) if y_all_ls else None
    else:
        s_all, y_all = _closure(base_dataset)
    return s_all, y_all


def get_group_ids(dataset: Dataset) -> Tensor:
    s_all, y_all = extract_labels_from_dataset(dataset)
    # group_ids: Optional[Tensor] = None
    if s_all is None:
        if y_all is None:
            raise ValueError(
                "Unable to compute group ids for dataset because no labels could be extracted."
            )
        group_ids = y_all
    elif y_all is None:
        group_ids = s_all
    else:
        group_ids = (y_all * len(s_all.unique()) + s_all).squeeze()
    return group_ids.long()


def compute_instance_weights(dataset: Dataset, *, upweight: bool = False) -> Tensor:
    group_ids = get_group_ids(dataset)
    _, inv_indexes, counts = group_ids.unique(return_inverse=True, return_counts=True)
    # Upweight samples according to the cardinality of their intersectional group
    if upweight:
        group_weights = len(group_ids) / counts
    # Downweight samples according to the cardinality of their intersectional group
    # - this approach should be preferred due to being more numerically stable
    # (very small counts can lead to very large weighted loss values when upweighting)
    else:
        group_weights = 1 - (counts / len(group_ids))
    return group_weights[inv_indexes]


PCD = TypeVar("PCD", bound=PseudoCdtDataset)


def make_subset(
    dataset: Union[PCD, Subset[PCD]],
    *,
    indices: Optional[Union[List[int], npt.NDArray[np.uint64], Tensor, slice]],
    deep: bool = False,
) -> PCD:
    """Create a subset of the dataset from the given indices.

    :param indices: The sample-indices from which to create the subset.
    In the case of being a numpy array or tensor, said array or tensor
    must be 0- or 1-dimensional.

    :param deep: Whether to create a copy of the underlying dataset as
    a basis for the subset. If False then the data of the subset will be
    a view of original dataset's data.

    :returns: A subset of the dataset from the given indices.
    """
    if isinstance(indices, (np.ndarray, Tensor)):
        if indices.ndim > 1:
            raise ValueError("If 'indices' is an array it must be a 0- or 1-dimensional.")
        indices = cast(List[int], indices.tolist())

    current_indices = None
    if isinstance(dataset, Subset):
        base_dataset, current_indices = extract_base_dataset(dataset, return_subset_indices=True)
        if not isinstance(base_dataset, CdtDataset):
            raise TypeError(
                f"Subsets can only be created from {CdtDataset.__name__} instances or PyTorch "
                "Subsets of them."
            )
        base_dataset = cast(PCD, base_dataset)

        if isinstance(current_indices, Tensor):
            current_indices = current_indices.tolist()
    else:
        base_dataset = dataset
    subset = gcopy(base_dataset, deep=deep)

    def _subset_from_indices(_dataset: PCD, _indices: Union[List[int], slice]) -> PCD:
        _dataset.x = _dataset.x[_indices]
        if _dataset.y is not None:
            _dataset.y = _dataset.y[_indices]
        if _dataset.s is not None:
            _dataset.s = _dataset.s[_indices]
        return _dataset

    if current_indices is not None:
        subset = _subset_from_indices(_dataset=subset, _indices=current_indices)
    if indices is not None:
        subset = _subset_from_indices(_dataset=subset, _indices=indices)

    return subset


def infer_sample_cls(
    sample: Union[List[LoadedData], Tuple[LoadedData, ...], Dict[str, LoadedData], LoadedData]
) -> Type[NamedSample]:
    """ "Attempt to infer the appropriate sample class based on the length of the input."""
    if not isinstance(sample, (list, tuple, dict)) or (len(sample) == 1):
        return NamedSample
    elif len(sample) == 2:
        return BinarySample
    elif len(sample) == 3:
        return TernarySample
    else:
        raise ValueError("Only items with 3 or fewer elements can be cast to 'Sample' instances.")


class cdt_collate:
    def __init__(
        self, cast_to_sample: bool = True, *, converter: Optional[Union[Type[Any], Callable]] = None
    ) -> None:
        self.cast_to_sample = cast_to_sample
        self.converter = converter

    def _collate(self, batch: Any) -> Any:
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:  # type: ignore
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            ndims = elem.dim()
            if (ndims > 0) and ((ndims % 2) == 0):
                return torch.cat(batch, dim=0, out=out)  # type: ignore
            return torch.stack(batch, dim=0, out=out)  # type: ignore
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

    def __call__(self, batch: Any) -> Any:
        collated_batch = self._collate(batch=batch)
        if self.converter is not None:
            collated_batch = self.converter(collated_batch)
        if self.cast_to_sample and (not isinstance(collated_batch, SampleBase)):
            if isinstance(collated_batch, Tensor):
                collated_batch = NamedSample(x=collated_batch)
            elif isinstance(collated_batch, (tuple, list, dict)):
                sample_cls = infer_sample_cls(collated_batch)
                if not isinstance(collated_batch, dict):
                    collated_batch = dict(zip(["x", "y", "s"], collated_batch))
                collated_batch = sample_cls(**collated_batch)
            else:
                raise ValueError(
                    f"batch of type '{type(collated_batch)}' could not be automatically cast to a "
                    "'Sample' instance. Batch must be of type 'dict', 'tuple', or 'list'."
                )
        return collated_batch


I = TypeVar("I", bound=NamedSample)


class CdtDataLoader(DataLoader[I]):
    def __init__(
        self,
        dataset: SizedDataset[I],
        *,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context: Optional[Union[BaseContext, str]] = None,
        generator: Optional[torch.Generator] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        cast_to_sample: bool = True,
        converter: Optional[Union[Type[Any], Callable]] = None,
    ) -> None:
        super().__init__(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=cdt_collate(cast_to_sample=cast_to_sample, converter=converter),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def __iter__(self) -> Iterator[I]:
        return super().__iter__()


def check_integrity(*, filepath: Path, md5: Optional[str]) -> None:
    from torchvision.datasets.utils import check_integrity  # type: ignore

    ext = filepath.suffix
    if ext not in [".zip", ".7z"] and check_integrity(fpath=str(filepath), md5=md5):
        raise RuntimeError('Dataset corrupted; try deleting it and redownloading it.')


class UrlFileInfo(NamedTuple):
    name: str
    url: str
    md5: Optional[str] = None


def download_from_url(
    *,
    file_info: Union[UrlFileInfo, List[UrlFileInfo]],
    root: Union[Path, str],
    logger: Optional[logging.Logger] = None,
    remove_finished: bool = True,
) -> None:

    logger = logging.getLogger(__name__) if logger is None else logger
    file_info_ls = file_info if isinstance(file_info, list) else [file_info]
    if not isinstance(root, Path):
        root = Path(root).expanduser()
    # Create the specified root directory if it doesn't already exist
    root.mkdir(parents=True, exist_ok=True)

    for info in file_info_ls:
        filepath = root / info.name

        filepath_str = str(filepath)
        suffix = _detect_file_type(filepath_str)[0]
        extracted_filepath = Path(filepath_str.split(suffix)[0])

        if extracted_filepath.exists():
            logger.info(f"File '{info.name}' already downloaded and extracted.")
        else:
            if filepath.exists():
                logger.info(f"File '{info.name}' already downloaded.")
            else:
                logger.info(f"Downloading file '{info.name}' from address '{info.url}'.")
                download_url(url=info.url, filename=info.name, root=str(root), md5=info.md5)

            logger.info(f"Extracting '{filepath.resolve()}' to '{root.resolve()}'")
            try:
                extract_archive(
                    from_path=str(filepath),
                    to_path=str(extracted_filepath),
                    remove_finished=remove_finished,
                )
            # Fall back on using jar to unzip the archive
            except BadZipFile:
                try:
                    subprocess.run(["jar", "-xvf", str(filepath)], check=True, cwd=root)
                except subprocess.CalledProcessError:
                    logger.info(
                        "Attempted to fall back on using Java to extract malformed .zip file; "
                        "however, there was a problem. Try redownloading the zip file or "
                        "checking that Java has been properly added to your system variables."
                    )


class GdriveFileInfo(NamedTuple):
    name: str
    id: str
    md5: Optional[str] = None


def download_from_gdrive(
    *,
    file_info: Union[GdriveFileInfo, List[GdriveFileInfo]],
    root: Union[Path, str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Attempt to download data if files cannot be found in the root directory."""

    logger = logging.getLogger(__name__) if logger is None else logger

    file_info_ls = file_info if isinstance(file_info, list) else [file_info]
    if not isinstance(root, Path):
        root = Path(root).expanduser()
    # Create the specified root directory if it doesn't already exist
    root.mkdir(parents=True, exist_ok=True)

    for info in file_info_ls:
        filepath = root / info.name
        if filepath.exists():
            logger.info(f"File '{info.name}' already downloaded.")
        else:
            import gdown  # type: ignore

            logger.info(f"Downloading file '{info.name}' from Google Drive.")
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={info.id}",
                path=str(filepath),
                quiet=False,
                md5=info.md5,
            )
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


def random_split(
    dataset: PCD,
    props: Union[Sequence[float], float],
    deep: bool = False,
    seed: Optional[int] = None,
) -> List[PCD]:
    """Randomly split the dataset into subsets according to the given proportions.

    :param props: The fractional size of each subset into which to randomly split the data.
    Elements must be non-negative and sum to 1 or less; if less then the size of the final
    split will be computed by complement.

    :param deep: Whether to create a copy of the underlying dataset as
    a basis for the random subsets. If False then the data of the subsets will be
    views of original dataset's data.

    :param seed: PRNG seed to use for sampling.

    :returns: Random subsets of the data of the requested proportions.
    """
    split_indices = prop_random_split(dataset=dataset, props=props, as_indices=True, seed=seed)
    splits = [make_subset(dataset, indices=indices, deep=deep) for indices in split_indices]
    return splits


def stratified_split(
    dataset: PCD,
    *,
    default_train_prop: float,
    train_props: Optional[Dict[int, Union[Dict[int, float], float]]] = None,
    seed: Optional[int] = None,
) -> TrainTestSplit[PCD]:
    """Splits the data into train/test sets conditional on super- and sub-class labels.

    :param default_train_prop: Proportion of samples for a given to sample for
    the training set for those y-s combinations not specified in ``train_props``.

    :param train_props: Proportion of each superclass-subclass combination to sample for
    the training set. Keys correspond to the superclass while values can be either a float,
    in which case the proportion is applied to the superclass as a whole, or a dict, in which
    case the sampling is applied only to the subclasses of the superclass given by the keys.
    If ``None`` then the function reduces to a simple random split of the data.

    :param seed: PRNG seed to use for sampling.

    :returns: Train-test split.
    """
    if dataset.y is None:
        raise TypeError(
            f"Dataset of type {dataset.__class__.__name__} has no superclass labels to use "
            "for stratification."
        )
    train_props = {} if train_props is None else train_props
    # Initialise the random-number generator
    generator = torch.default_generator if seed is None else torch.Generator().manual_seed(seed)

    group_ids = get_group_ids(dataset)
    y_unique = dataset.y.unique()
    groups, id_counts = group_ids.unique(return_counts=True)
    card_s = None if dataset.s is None else len(dataset.s.unique())
    ncols = 1 if card_s is None else card_s
    group_train_props = dict.fromkeys(groups.tolist(), default_train_prop)

    if train_props is not None:
        for superclass, value in train_props.items():
            # Apply the same splitting proportion to the entire superclass
            if superclass not in y_unique:
                raise ValueError(
                    f"No samples belonging to superclass 'y={superclass}' exist in the dataset of "
                    f"type {dataset.__class__.__name__}."
                )
            if isinstance(value, float):
                if not 0 <= value <= 1:
                    raise ValueError(
                        "All splitting proportions speicfied in 'train_props' must be in the "
                        "range [0, 1]."
                    )
                if card_s is None:
                    group_train_props[superclass] = value
                else:
                    superclass_props = dict.fromkeys(
                        range(superclass * card_s, (superclass + 1) * card_s),
                        value,
                    )
                    group_train_props.update(superclass_props)
            # Specifying proportions at the superclass/subclass level, rather than superclass-wide
            else:
                for subclass, train_prop in value.items():
                    if not 0 <= train_prop <= 1:
                        raise ValueError(
                            "All splitting proportions specified in 'train_props' must be in the "
                            "range [0, 1]."
                        )
                    group_id = superclass * ncols + subclass
                    if group_id not in groups:
                        raise ValueError(
                            f"No samples belonging to the subset '(y={superclass}', s={subclass})' "
                            f"exist in the dataset of type {dataset.__class__.__name__}."
                        )
                    group_train_props[group_id] = train_prop

    # Shuffle the samples before sampling
    perm_inds = torch.randperm(len(group_ids), generator=generator)
    group_ids_perm = group_ids[perm_inds]

    sort_inds = group_ids_perm.sort(dim=0, stable=True).indices
    thresholds = cast(
        Tensor, (torch.as_tensor(tuple(group_train_props.values())) * id_counts).round().long()
    )
    thresholds = torch.stack([thresholds, id_counts], dim=-1)
    thresholds[1:] += id_counts.cumsum(0)[:-1].unsqueeze(-1)

    train_test_inds = sort_inds.tensor_split(thresholds.flatten()[:-1], dim=0)
    train_inds = perm_inds[torch.cat(train_test_inds[0::2])]
    test_inds = perm_inds[torch.cat(train_test_inds[1::2])]

    train_data = make_subset(dataset=dataset, indices=train_inds)
    test_data = make_subset(dataset=dataset, indices=test_inds)

    return TrainTestSplit(train=train_data, test=test_data)


def is_tensor_list(ls: List[Any]) -> TypeGuard[List[Tensor]]:
    return isinstance(ls[0], Tensor)
