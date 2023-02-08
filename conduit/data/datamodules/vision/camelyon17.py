"""Camelyon17 data-module."""
from typing import Any

import albumentations as A  # type: ignore
import attr
from typing_extensions import override

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.vision.camelyon17 import (
    Camelyon17,
    Camelyon17Attr,
    Camelyon17Split,
    Camelyon17SplitScheme,
    SampleType,
)
from conduit.data.structures import TrainValTestSplit

__all__ = ["Camelyon17DataModule"]


@attr.define(kw_only=True)
class Camelyon17DataModule(CdtVisionDataModule[Camelyon17, SampleType]):
    """Data-module for the Camelyon17 dataset."""

    image_size: int = 96
    superclass: Camelyon17Attr = Camelyon17Attr.TUMOR
    subclass: Camelyon17Attr = Camelyon17Attr.CENTER
    use_predefined_splits: bool = False
    split_scheme: Camelyon17SplitScheme = Camelyon17SplitScheme.OFFICIAL

    @override
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Camelyon17(root=self.root, download=True)

    @property
    @override
    def _default_train_transforms(self) -> A.Compose:
        base_transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
                A.HorizontalFlip(),
                A.RandomCrop(self.image_size, self.image_size),
            ]
        )
        normalization = super()._default_train_transforms

        return A.Compose([base_transforms, normalization])

    @property
    @override
    def _default_test_transforms(self) -> A.Compose:
        base_transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
            ]
        )
        normalization = super()._default_train_transforms

        return A.Compose([base_transforms, normalization])

    @override
    def _get_splits(self) -> TrainValTestSplit[Camelyon17]:
        # Split the data according to the pre-defined split indices
        if self.use_predefined_splits:
            train_data, val_data, test_data = (
                Camelyon17(root=self.root, split=split, split_scheme=self.split_scheme)
                for split in Camelyon17Split
            )
        # Split the data randomly according to test- and val-prop
        else:
            all_data = Camelyon17(root=self.root, transform=None, split_scheme=self.split_scheme)
            val_data, test_data, train_data = all_data.random_split(
                props=(self.val_prop, self.test_prop), seed=self.seed
            )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
