"""Base class for vision datasets."""
from pathlib import Path
from typing import List, Optional, TypeVar, Union

import albumentations as A  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
import attr
from typing_extensions import final, override

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.base import I
from conduit.data.datasets.vision import (
    AlbumentationsTform,
    ImageTform,
    ImageTransformer,
)
from conduit.data.datasets.wrappers import InstanceWeightedDataset
from conduit.data.structures import ImageSize, MeanStd, SizedDataset
from conduit.types import Stage

__all__ = ["CdtVisionDataModule"]

D = TypeVar("D", bound=SizedDataset)


@attr.define(kw_only=True)
class CdtVisionDataModule(CdtDataModule[D, I]):
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

    @override
    def _setup(self, stage: Optional[Stage] = None) -> None:
        train, val, test = self._get_splits()
        train = ImageTransformer(train, transform=self.train_transforms)
        if self.instance_weighting:
            train = InstanceWeightedDataset(train)
        self._train_data = train
        self._val_data = ImageTransformer(val, transform=self.test_transforms)
        self._test_data = ImageTransformer(test, transform=self.test_transforms)
