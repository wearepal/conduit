"""CelebA DataModule."""
from typing import Any, Optional

import albumentations as A
from kit import implements, parsable
from kit.torch import prop_random_split
from pytorch_lightning import LightningDataModule

from bolts.data.datamodules.common import Stage
from bolts.data.datasets.vision.nico import NICO, NicoSuperclass
from bolts.data.datasets.wrappers import ImageTransformer

from .base import VisionDataModule

__all__ = ["NICODataModule"]


class NICODataModule(VisionDataModule):
    """Data-module for the NICO dataset."""

    @parsable
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        class_train_props: Optional[dict] = None,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        superclass: NicoSuperclass = NicoSuperclass.animals,
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
        self.class_train_props = class_train_props

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        NICO(root=self.root, download=True)

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
        return A.Compose(
            [
                A.RandomResizedCrop(height=self.image_size, width=self.image_size),
                A.HorizontalFlip(p=0.5),
            ]
        )

    @implements(LightningDataModule)
    def setup(self, stage: Optional[Stage] = None) -> None:
        all_data = NICO(root=self.root, superclass=self.superclass, transform=None)

        train_val_data, test_data = all_data.train_test_split(
            default_train_prop=1 - self.test_prop,
            train_props=self.class_train_props,
            seed=self.seed,
        )
        val_data, train_data = prop_random_split(dataset=train_val_data, props=self.val_prop)
        self._train_data = ImageTransformer(train_data, transform=self._augmentations(train=True))
        self._val_data = ImageTransformer(val_data, transform=self._augmentations(train=False))
        self._test_data = ImageTransformer(test_data, transform=self._augmentations(train=False))
