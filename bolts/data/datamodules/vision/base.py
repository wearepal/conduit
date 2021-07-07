from __future__ import annotations
from abc import abstractmethod
import logging
from typing import ClassVar

import albumentations as A
from albumentations.pytorch import ToTensorV2

from bolts.data.datamodules import BaseDataModule
from bolts.data.structures import InputSize, NormalizationValues

__all__ = ["VisionDataModule", "TrainAugMode"]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class VisionDataModule(BaseDataModule):
    _input_size: InputSize
    norm_values: ClassVar[NormalizationValues] = NormalizationValues(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 0,
        persist_workers: bool = False,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            seed=seed,
            test_prop=test_prop,
            val_prop=val_prop,
        )
        self.root = root

    @property
    def input_size(self) -> InputSize:
        if hasattr(self, "_input_size"):
            return self._input_size
        if hasattr(self, "_train_data"):
            self._input_size = InputSize(*self._train_data[0].x.shape)
            return self._input_size
        raise AttributeError("Input size unavailable because setup has not yet been called.")

    @property
    @abstractmethod
    def _base_augmentations(self) -> A.Compose:
        ...

    @property
    @abstractmethod
    def _train_augmentations(self) -> A.Compose:
        ...

    @property
    def _normalization(self) -> A.Compose:
        return A.Compose(
            [
                A.ToFloat(),
                A.Normalize(mean=self.norm_values.mean, std=self.norm_values.std),
                ToTensorV2(),
            ]
        )

    def _augmentations(self, train: bool) -> A.Compose:
        # Base augmentations (augmentations that are applied to all splits of the data)
        augs: list[A.ImageOnlyTransform | A.Compose] = [self._base_augmentations]
        # Add training augmentations on top of base augmentations
        if train:
            augs.append(self._train_augmentations)
        # Normalization is common to all splits but needs to be applied at the end of the
        # transformation pipeline.
        augs.append(self._normalization)
        return A.Compose(augs)
