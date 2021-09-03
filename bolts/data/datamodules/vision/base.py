"""Base class for vision datasets."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
from kit import implements
from kit.torch import TrainingMode
import pytorch_lightning as pl
from typing_extensions import final

from bolts.constants import IMAGENET_STATS
from bolts.data.datamodules import PBDataModule
from bolts.data.datasets.utils import AlbumentationsTform, ImageTform
from bolts.data.datasets.wrappers import ImageTransformer, InstanceWeightedDataset
from bolts.data.structures import ImageSize, MeanStd
from bolts.types import Stage

__all__ = ["PBVisionDataModule"]


class PBVisionDataModule(PBDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        train_batch_size: int = 64,
        eval_batch_size: Optional[int] = 100,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = "epoch",
        train_transforms: ImageTform | None = None,
        test_transforms: ImageTform | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            seed=seed,
            test_prop=test_prop,
            val_prop=val_prop,
            stratified_sampling=stratified_sampling,
            instance_weighting=instance_weighting,
            training_mode=training_mode,
        )

        pl.LightningDataModule.__init__(
            self, train_transforms=train_transforms, test_transforms=test_transforms
        )
        self.root = root
        self.norm_values: MeanStd | None = IMAGENET_STATS
        self._input_size: ImageSize | None = None

    @property
    @final
    def size(self) -> ImageSize:
        if self._input_size is not None:
            return self._input_size
        if self._train_data is not None:
            self._input_size = ImageSize(*self._train_data[0].x.shape[-3:])  # type: ignore
            return self._input_size
        cls_name = self.__class__.__name__
        raise AttributeError(
            f"'{cls_name}.size' cannot be determined because 'setup' has not yet been called."
        )

    @property
    @final
    def train_transforms(self) -> ImageTform:
        return (
            self._default_train_transforms
            if self._train_transforms is None
            else self._train_transforms
        )

    @train_transforms.setter
    def train_transforms(self, transform: ImageTform | None) -> None:  # type: ignore
        self._train_transforms = transform
        if isinstance(self._train_data, ImageTransformer):
            self._train_data.transform = transform

    @property
    @final
    def test_transforms(self) -> ImageTform:
        return (
            self._default_test_transforms
            if self._test_transforms is None
            else self._test_transforms
        )

    @test_transforms.setter
    @final
    def test_transforms(self, transform: ImageTform | None) -> None:  # type: ignore
        self._test_transforms = transform
        if isinstance(self._val_data, ImageTransformer):
            self._val_data.transform = transform
        if isinstance(self._test_data, ImageTransformer):
            self._test_data.transform = transform

    @property
    def _default_train_transforms(self) -> A.Compose:
        transform_ls: list[AlbumentationsTform] = [
            A.ToFloat(),
        ]
        if self.norm_values is not None:
            transform_ls.append(A.Normalize(mean=self.norm_values.mean, std=self.norm_values.std))
        transform_ls.append(ToTensorV2())
        return A.Compose(transform_ls)

    @property
    def _default_test_transforms(self) -> A.Compose:
        transform_ls: list[AlbumentationsTform] = [
            A.ToFloat(),
        ]
        if self.norm_values is not None:
            transform_ls.append(A.Normalize(mean=self.norm_values.mean, std=self.norm_values.std))
        transform_ls.append(ToTensorV2())
        return A.Compose(transform_ls)

    @implements(PBDataModule)
    @final
    def _setup(self, stage: Stage | None = None) -> None:
        train, val, test = self._get_splits()
        train = ImageTransformer(train, transform=self.train_transforms)
        if self.instance_weighting:
            train = InstanceWeightedDataset(train)
        self._train_data = train
        self._val_data = ImageTransformer(val, transform=self.test_transforms)
        self._test_data = ImageTransformer(test, transform=self.test_transforms)
