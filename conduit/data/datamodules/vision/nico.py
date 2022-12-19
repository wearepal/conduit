"""NICO data-module."""
from typing import Any, Optional

import albumentations as A  # type: ignore
import attr
from typing_extensions import override

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import stratified_split
from conduit.data.datasets.vision.nico import NICO, NicoSuperclass, SampleType
from conduit.data.structures import TrainValTestSplit

__all__ = ["NICODataModule"]


@attr.define(kw_only=True)
class NICODataModule(CdtVisionDataModule[NICO, SampleType]):
    """Data-module for the NICO dataset."""

    image_size: int = 224
    class_train_props: Optional[dict] = None
    superclass: NicoSuperclass = NicoSuperclass.ANIMALS

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
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        NICO(root=self.root, download=True)

    @override
    def _get_splits(self) -> TrainValTestSplit[NICO]:
        all_data = NICO(root=self.root, superclass=self.superclass, transform=None)
        train_val_prop = 1 - self.test_prop
        train_val_data, test_data = stratified_split(
            all_data,
            default_train_prop=train_val_prop,
            train_props=self.class_train_props,
            seed=self.seed,
        )
        val_data, train_data = train_val_data.random_split(
            props=self.val_prop / train_val_prop, seed=self.seed
        )

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
