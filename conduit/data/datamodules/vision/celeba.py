"""CelebA data-module."""
from typing import Any

import albumentations as A  # type: ignore
import attr
from typing_extensions import override

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.vision.celeba import (
    CelebA,
    CelebASplit,
    CelebAttr,
    SampleType,
)
from conduit.data.structures import TrainValTestSplit

__all__ = ["CelebADataModule"]


@attr.define(kw_only=True)
class CelebADataModule(CdtVisionDataModule[CelebA, SampleType]):
    """Data-module for the CelebA dataset."""

    image_size: int = 224
    superclass: CelebAttr = CelebAttr.SMILING
    subclass: CelebAttr = CelebAttr.MALE
    use_predefined_splits: bool = False

    @override
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        CelebA(root=self.root, download=True)

    @property
    @override
    def _default_train_transforms(self) -> A.Compose:
        base_transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
            ]
        )
        normalization = super()._default_train_transforms
        return A.Compose([base_transforms, normalization])

    @property
    @override
    def _default_test_transforms(self) -> A.Compose:
        return self._default_train_transforms

    @override
    def _get_splits(self) -> TrainValTestSplit[CelebA]:
        # Split the data according to the pre-defined split indices
        if self.use_predefined_splits:
            train_data, val_data, test_data = (
                CelebA(root=self.root, superclass=self.superclass, transform=None, split=split)
                for split in CelebASplit
            )
        # Split the data randomly according to test- and val-prop
        else:
            all_data = CelebA(root=self.root, superclass=self.superclass, transform=None)
            val_data, test_data, train_data = all_data.random_split(
                props=(self.val_prop, self.test_prop),
                seed=self.seed,
            )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
