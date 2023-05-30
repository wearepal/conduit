"""Base class for vision datasets."""
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, TypeVar, Union

import albumentations as A  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
import attr
from typing_extensions import final, override

from torch import Tensor

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.base import I
from conduit.data.datasets.vision import (
    AlbumentationsTform,
    ImageTform,
    ImageTransformer,
)
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import ImageSize, MeanStd, SizedDataset, TrainValTestSplit

__all__ = ["CdtVisionDataModule"]

D = TypeVar("D", bound=SizedDataset)


@attr.define(kw_only=True)
class CdtVisionDataModule(CdtDataModule[ImageTransformer, I]):
    root: Union[str, Path] = attr.field(kw_only=False)
    _train_transforms: Optional[ImageTform] = None
    _test_transforms: Optional[ImageTform] = None
    norm_values: Optional[MeanStd] = attr.field(default=IMAGENET_STATS, init=False)

    @property
    @final
    def train_transforms(self) -> ImageTform:
        return (
            self._default_train_transforms
            if self._train_transforms is None
            else self._train_transforms
        )

    @train_transforms.setter
    def train_transforms(self, transform: Optional[ImageTform]) -> None:
        self._train_transforms = transform
        if isinstance(self._train_data, ImageTransformer):
            self._train_data.transform = transform

    @property
    @final
    def test_transforms(self) -> ImageTform:
        return (
            self._default_test_transforms
            if self._test_transforms is None
            else self._test_transforms
        )

    @test_transforms.setter
    @final
    def test_transforms(self, transform: Optional[ImageTform]) -> None:
        self._test_transforms = transform
        if isinstance(self._val_data, ImageTransformer):
            self._val_data.transform = transform
        if isinstance(self._test_data, ImageTransformer):
            self._test_data.transform = transform

    @property
    def _default_train_transforms(self) -> A.Compose:
        transform_ls: List[AlbumentationsTform] = [A.ToFloat()]
        if self.norm_values is not None:
            # `max_pixel_value` has to be 1.0 here because of `ToFloat()`
            transform_ls.append(
                A.Normalize(
                    mean=self.norm_values.mean, std=self.norm_values.std, max_pixel_value=1.0
                )
            )
        transform_ls.append(ToTensorV2())
        return A.Compose(transform_ls)

    @property
    @override
    def dim_x(self) -> ImageSize:
        """
        Returns the dimensions of the first input (x).

        :returns: ImageSize object containing the dimensions (C, H, W) of the (first) input.
        """
        return ImageSize(*super().dim_x)

    @override
    def size(self) -> ImageSize:
        """Alias for ``dim_x``.

        :returns: ImageSize object containing the dimensions (C, H, W) of the (first) input.
        """
        return self.dim_x

    @property
    def _default_test_transforms(self) -> A.Compose:
        transform_ls: List[AlbumentationsTform] = [A.ToFloat()]
        if self.norm_values is not None:
            transform_ls.append(A.Normalize(mean=self.norm_values.mean, std=self.norm_values.std))
        transform_ls.append(ToTensorV2())
        return A.Compose(transform_ls)

    @abstractmethod
    def _get_image_splits(self) -> TrainValTestSplit[CdtVisionDataset[I, Tensor, Tensor]]:
        ...

    @override
    def _get_splits(self) -> TrainValTestSplit[ImageTransformer]:
        train, val, test = self._get_image_splits()
        return TrainValTestSplit(
            train=ImageTransformer(train, transform=self.train_transforms),
            val=ImageTransformer(val, transform=self.test_transforms),
            test=ImageTransformer(test, transform=self.test_transforms),
        )
