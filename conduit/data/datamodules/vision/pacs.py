"""PACS datamodule."""
from typing import Any

import albumentations as A  # type: ignore
import attr
from typing_extensions import override

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.vision.pacs import PACS, SampleType
from conduit.data.structures import TrainValTestSplit

__all__ = ["PACSDataModule"]


@attr.define(kw_only=True)
class PACSDataModule(CdtVisionDataModule[PACS, SampleType]):
    """PyTorch Lightning Datamodule for the PACS dataset."""

    image_size: int = 224
    target_domain: PACS.Domain = PACS.Domain.SKETCH

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
        PACS(root=self.root, download=True)

    @override
    def _get_splits(self) -> TrainValTestSplit[PACS]:
        all_data = PACS(root=self.root, domains=None, transform=None)
        train_val_data, test_data = all_data.domain_split(target_domains=self.target_domain)
        val_data, train_data = train_val_data.random_split(props=self.val_prop, seed=self.seed)

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
