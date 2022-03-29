"""CelebA data-module."""
from typing import Any

import albumentations as A  # type: ignore
import attr
from pytorch_lightning import LightningDataModule
from ranzen import implements

from conduit.data.datamodules.base import CdtDataModule
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
    superclass: CelebAttr = CelebAttr.Smiling
    subclass: CelebAttr = CelebAttr.Male
    use_predefined_splits: bool = False

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        CelebA(root=self.root, download=True)

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_train_transforms(self) -> A.Compose:
        base_transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
            ]
        )
        normalization = super()._default_train_transforms
        return A.Compose([base_transforms, normalization])

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_test_transforms(self) -> A.Compose:
        return self._default_train_transforms

    @implements(CdtDataModule)
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
