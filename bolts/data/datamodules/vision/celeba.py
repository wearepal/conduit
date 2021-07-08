"""CelebA data-module."""
from typing import Any, Optional

import albumentations as A
from kit import implements, parsable
from kit.torch import prop_random_split
from pytorch_lightning import LightningDataModule

from bolts.common import Stage
from bolts.data.datasets.vision.celeba import CelebA, CelebASplit, CelebAttr
from bolts.data.datasets.wrappers import ImageTransformer

from .base import VisionDataModule

__all__ = ["CelebADataModule"]


class CelebADataModule(VisionDataModule):
    """Data-module for the CelebA dataset."""

    @parsable
    def __init__(
        self,
        root: str,
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
        )
        self.image_size = image_size
        self.superclass = superclass
        self.subclass = subclass
        self.use_predefined_splits = use_predefined_splits

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        CelebA(root=self.root, download=True)

    @property  # type: ignore[misc]
    @implements(VisionDataModule)
    def _base_augmentations(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
            ]
        )

    @property  # type: ignore[misc]
    @implements(VisionDataModule)
    def _train_augmentations(self) -> A.Compose:
        return A.Compose([])

    @implements(LightningDataModule)
    def setup(self, stage: Optional[Stage] = None) -> None:
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
        self._train_data = ImageTransformer(train_data, transform=self._augmentations(train=True))
        self._val_data = ImageTransformer(val_data, transform=self._augmentations(train=False))
        self._test_data = ImageTransformer(test_data, transform=self._augmentations(train=False))
