"""NICO++ data-module."""

from dataclasses import dataclass
from typing import Any, List, Optional
from typing_extensions import override

import albumentations as A  # type: ignore

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.vision import NICOPP, NicoPPSplit, NicoPPTarget, SampleType
from conduit.data.structures import TrainValTestSplit

__all__ = ["NICOPPDataModule"]


@dataclass(kw_only=True)
class NICOPPDataModule(CdtVisionDataModule[SampleType]):
    """Data-module for the NICO dataset."""

    image_size: int = 224
    superclasses: Optional[List[NicoPPTarget]] = None

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
    def _get_image_splits(self) -> TrainValTestSplit[NICOPP]:
        train_data, val_data, test_data = (
            NICOPP(root=self.root, superclasses=self.superclasses, transform=None, split=split)
            for split in NicoPPSplit
        )

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
