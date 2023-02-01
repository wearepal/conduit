"""Waterbirds data-module."""
from typing import Any

import albumentations as A  # type: ignore
import attr
from typing_extensions import override

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.vision.waterbirds import Waterbirds
from conduit.data.structures import TrainValTestSplit

__all__ = ["WaterbirdsDataModule"]


@attr.define(kw_only=True)
class WaterbirdsDataModule(CdtVisionDataModule[Waterbirds, Waterbirds.SampleType]):
    """Data-module for the Waterbirds dataset."""

    image_size: int = 224
    use_predefined_splits: bool = False

    @override
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Waterbirds(root=self.root, download=True)

    @property
    @override
    def _default_train_transforms(self) -> A.Compose:
        # We use the transoform pipeline described in https://arxiv.org/abs/2008.06775
        # rather than that described in the paper in which the dataset was first introduced
        # (Sagawa et al); these differ in in the respect that the latter center-crops the images
        # before resizing - not doing so makes the task more difficult due to the background
        # (serving as a spurious attribute) being more prominent.
        base_transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
            ]
        )
        normalization = super()._default_train_transforms
        return A.Compose([base_transforms, normalization])

    @property
    @override
    def _default_test_transforms(self) -> A.Compose:
        return self._default_train_transforms

    @override
    def _get_splits(self) -> TrainValTestSplit[Waterbirds]:
        # Split the data according to the pre-defined split indices
        if self.use_predefined_splits:
            train_data, val_data, test_data = (
                Waterbirds(root=self.root, split=split) for split in Waterbirds.Split
            )
        # Split the data randomly according to test- and val-prop
        else:
            all_data = Waterbirds(root=self.root, transform=None)
            val_data, test_data, train_data = all_data.random_split(
                props=(self.val_prop, self.test_prop), seed=self.seed
            )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
