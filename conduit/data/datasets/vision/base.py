from functools import reduce
import operator
from pathlib import Path
from typing import List, Optional, Sequence, Union, cast, overload

import numpy as np
import numpy.typing as npt
from ranzen.types import Addable
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias, override

from conduit.data.datasets.base import CdtDataset, I, S, Y
from conduit.data.datasets.vision.utils import (
    ImageLoadingBackend,
    ImageTform,
    RawImage,
    apply_image_transform,
    img_to_tensor,
    infer_il_backend,
    load_image,
)
from conduit.data.structures import TargetData
from conduit.types import IndexType

__all__ = ["CdtVisionDataset"]

ItemType: TypeAlias = Union[RawImage, Tensor, Sequence[RawImage], Sequence[Tensor]]


class CdtVisionDataset(CdtDataset[I, npt.NDArray[np.string_], Y, S]):
    def __init__(
        self,
        *,
        x: npt.NDArray[np.string_],
        image_dir: Union[Path, str],
        y: Optional[TargetData] = None,
        s: Optional[TargetData] = None,
        transform: Optional[ImageTform] = None,
    ) -> None:
        super().__init__(x=x, y=y, s=s)
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        self.image_dir = image_dir
        self.transform = transform
        # infer the appropriate image-loading backend based on the type of 'transform'
        self._il_backend: ImageLoadingBackend = infer_il_backend(transform)

    def update_il_backend(self, transform: Optional[ImageTform]) -> None:
        new_backend = infer_il_backend(transform)
        if new_backend != self._il_backend:
            self._il_backend = new_backend
            self.logger.info(f"Using {self._il_backend} as backend for image-loading.")

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {len(self)}",
            f"Base image-directory location: {self.image_dir.resolve()}",
            *self.extra_repr().splitlines(),
        ]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _load_image(self, index: IndexType) -> RawImage:
        filepath = cast(str, self.x[index])
        return load_image(self.image_dir / filepath, backend=self._il_backend)

    @overload
    def _sample_x(self, index: int, *, coerce_to_tensor: bool = ...) -> ItemType:
        ...

    @overload
    def _sample_x(self, index: List[int], *, coerce_to_tensor: bool = ...) -> List[ItemType]:
        ...

    @override
    def _sample_x(
        self, index: IndexType, *, coerce_to_tensor: bool = False
    ) -> Union[ItemType, List[ItemType]]:
        if isinstance(index, slice):
            index = list(range(len(self)))[index]
        if isinstance(index, list):
            sample_ls = [self._sample_x(index=i, coerce_to_tensor=coerce_to_tensor) for i in index]
            elem = sample_ls[0]
            if isinstance(elem, Addable):
                summed = reduce(operator.add, sample_ls)
                return cast(ItemType, summed)
            if isinstance(elem, Tensor):
                sample_ls = cast(List[Tensor], sample_ls)
                return torch.stack(sample_ls, dim=0)
            elif isinstance(sample_ls[0], np.ndarray):
                return np.stack(sample_ls, axis=0)
            return sample_ls

        image = self._load_image(index)
        image = apply_image_transform(image=image, transform=self.transform)
        if coerce_to_tensor and (not isinstance(image, Tensor)):
            if isinstance(image, Sequence):
                image = [img_to_tensor(subimage) for subimage in image]
            else:
                image = img_to_tensor(image)
        return image

    @override
    def subset(
        self,
        indices: Union[List[int], npt.NDArray[np.uint64], Tensor, slice],
        deep: bool = False,
        transform: Optional[ImageTform] = None,
    ) -> Self:
        """Create a subset of the dataset from the given indices.

        :param indices: The sample-indices from which to create the subset. In the case of being a
            numpy array or tensor, said array or tensor must be 0- or 1-dimensional.

        :param deep: Whether to create a copy of the underlying dataset as a basis for the subset.
            If False then the data of the subset will be a view of original dataset's data.

        :param transform: Image transform to assign to the resulting subset.

        :returns: A subset of the dataset from the given indices.
        """
        subset = super().subset(indices=indices, deep=deep)
        if transform is not None:
            subset.transform = transform
        return subset
