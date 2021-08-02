"""CelebA data-module."""
from typing import Any, Union

import albumentations as A
from kit import implements, parsable
from kit.torch import TrainingMode, prop_random_split
from pytorch_lightning import LightningDataModule

from bolts.data.datamodules.base import PBDataModule
from bolts.data.datasets.vision.celeba import CelebA, CelebASplit, CelebAttr
from bolts.data.structures import TrainValTestSplit

from .base import PBVisionDataModule

__all__ = ["CelebADataModule"]


class CelebADataModule(PBVisionDataModule):
    """Data-module for the CelebA dataset."""

    @parsable
    def __init__(
        self,
        root: str,
        *,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        superclass: CelebAttr = CelebAttr.Smiling,
        subclass: CelebAttr = CelebAttr.Male,
        use_predefined_splits: bool = False,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = TrainingMode.epoch,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_prop=val_prop,
            test_prop=test_prop,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            stratified_sampling=stratified_sampling,
            instance_weighting=instance_weighting,
            training_mode=training_mode,
        )
        self.image_size = image_size
        self.superclass = superclass
        self.subclass = subclass
        self.use_predefined_splits = use_predefined_splits

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        CelebA(root=self.root, download=True)

    @property  # type: ignore[misc]
    @implements(PBVisionDataModule)
    def _base_augmentations(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
            ]
        )

    @property  # type: ignore[misc]
    @implements(PBVisionDataModule)
    def _train_augmentations(self) -> A.Compose:
        return A.Compose([])

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # Split the data according to the pre-defined split indices
        if self.use_predefined_splits:
            train_data, val_data, test_data = (
                CelebA(root=self.root, superclass=self.superclass, transform=None, split=split)
                for split in CelebASplit
            )
        # Split the data randomly according to test- and val-prop
        else:
            all_data = CelebA(root=self.root, superclass=self.superclass, transform=None)
            val_data, test_data, train_data = prop_random_split(
                dataset=all_data, props=(self.val_prop, self.test_prop)
            )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
