"""Nico data-module."""
from typing import Any, Optional, Union

import albumentations as A
from kit import implements, parsable
from kit.torch import prop_random_split
from kit.torch.data import TrainingMode
from pytorch_lightning import LightningDataModule

from bolts.data.datamodules import PBDataModule
from bolts.data.datasets.vision.nico import NICO, NicoSuperclass
from bolts.data.structures import TrainValTestSplit

from .base import PBVisionDataModule

__all__ = ["NICODataModule"]


class NICODataModule(PBVisionDataModule):
    """Data-module for the NICO dataset."""

    @parsable
    def __init__(
        self,
        root: str,
        *,
        image_size: int = 224,
        train_batch_size: int = 32,
        eval_batch_size: Optional[int] = 64,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        class_train_props: Optional[dict] = None,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        superclass: NicoSuperclass = NicoSuperclass.animals,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = TrainingMode.epoch,
    ) -> None:
        super().__init__(
            root=root,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
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
        self.class_train_props = class_train_props

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

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        NICO(root=self.root, download=True)

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        all_data = NICO(root=self.root, superclass=self.superclass, transform=None)

        train_val_prop = 1 - self.test_prop
        train_val_data, test_data = all_data.train_test_split(
            default_train_prop=train_val_prop,
            train_props=self.class_train_props,
            seed=self.seed,
        )
        val_data, train_data = prop_random_split(
            dataset=train_val_data, props=self.val_prop / train_val_prop
        )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
