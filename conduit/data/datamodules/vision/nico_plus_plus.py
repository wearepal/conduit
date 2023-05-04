"""NICO++ data-module."""
from typing import Any, List, Optional

import albumentations as A  # type: ignore
import attr
from typing_extensions import override

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import stratified_split
from conduit.data.datasets.vision import NICOPP, NicoPPTarget, SampleType
from conduit.data.structures import TrainValTestSplit

__all__ = ["NICOPPDataModule"]


@attr.define(kw_only=True)
class NICOPPDataModule(CdtVisionDataModule[NICOPP, SampleType]):
    """Data-module for the NICO dataset."""

    image_size: int = 224
    superclasses: Optional[List[NicoPPTarget]] = None
    make_biased: bool = True

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
        pass

    @override
    def _get_splits(self) -> TrainValTestSplit[NICOPP]:
        all_data = NICOPP(root=self.root, superclasses=self.superclasses, transform=None)
        train_val_prop = 1 - self.test_prop
        train_val_data, test_data = stratified_split(
            all_data,
            default_train_prop=train_val_prop,
            train_props=all_data.default_train_props if self.make_biased else None,
            seed=self.seed,
        )
        val_data, train_data = train_val_data.random_split(
            props=self.val_prop / train_val_prop, seed=self.seed
        )

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
