"""ColoredMNIST data-module."""
from __future__ import annotations
from functools import partial
from typing import Any, Dict, List, Optional, Union

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from kit import implements, parsable
from kit.torch import TrainingMode, prop_random_split
import numpy as np
from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST

from bolts.data.datamodules import PBDataModule
from bolts.data.datasets.utils import ImageTform
from bolts.data.datasets.vision.cmnist import ColoredMNIST, ColoredMNISTSplit
from bolts.data.structures import MeanStd, TrainValTestSplit

from .base import PBVisionDataModule

__all__ = ["ColoredMNISTDataModule"]


class ColoredMNISTDataModule(PBVisionDataModule):
    """Data-module for the ColoredMNIST dataset."""

    @parsable
    def __init__(
        self,
        root: str,
        *,
        image_size: int = 32,
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
        use_predefined_splits: bool = False,
        # ColoredMNIST settings
        label_map: Optional[Dict[str, int]] = None,
        colors: Optional[List[int]] = None,
        num_colors: int = 10,
        scale: float = 0.2,
        correlation: float = 1.0,
        binarize: bool = False,
        greyscale: bool = False,
        background: bool = False,
        black: bool = True,
        train_transforms: Optional[ImageTform] = None,
        test_transforms: Optional[ImageTform] = None,
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
            train_transforms=train_transforms,
            test_transforms=test_transforms,
        )
        self.image_size = image_size
        self.use_predefined_splits = use_predefined_splits
        self.label_map = label_map
        self.colors = colors
        self.num_colors = num_colors
        self.scale = scale
        self.correlation = correlation
        self.binarize = binarize
        self.greyscale = greyscale
        self.background = background
        self.black = black

    @property  # type: ignore[misc]
    @implements(PBVisionDataModule)
    def _default_train_transforms(self) -> A.Compose:
        base_transforms = A.Compose([A.Resize(self.image_size, self.image_size)])
        normalization = A.Compose([A.Normalize(self.norm_values), ToTensorV2])
        return A.Compose([base_transforms, normalization])

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        MNIST(root=str(self.root), download=True, train=True)
        MNIST(root=str(self.root), download=True, train=False)

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # TODO: Add more sophisticated (e.g. biased) splits
        fact_func = partial(
            ColoredMNIST,
            root=self.root,
            background=self.background,
            black=self.black,
            greyscale=self.greyscale,
            correlation=self.correlation,
            colors=self.colors,
            num_colors=self.num_colors,
            label_map=self.label_map,
        )
        # Use the predefined train and test splits, sampling the val split
        # randomly from the train split
        if self.use_predefined_splits:
            train_data = fact_func(split=ColoredMNISTSplit.train)
            test_data = fact_func(split=ColoredMNISTSplit.test)
            val_data, train_data_new = prop_random_split(dataset=train_data, props=self.val_prop)
        else:
            # Split the data randomly according to val- and test-prop
            train_data = fact_func(split=None)
            val_data, test_data, train_data_new = prop_random_split(
                dataset=train_data, props=(self.val_prop, self.test_prop)
            )
        # Compute the channel-wise first and second moments
        channel_means = np.mean(train_data.x[train_data_new.indices], axis=(0, 1, 2)) / 255.0
        channel_stds = np.std(train_data.x[train_data_new.indices], axis=(0, 1, 2)) / 255.0

        self.norm_values = MeanStd(mean=channel_means.tolist(), std=channel_stds.tolist())

        return TrainValTestSplit(train=train_data_new, val=val_data, test=test_data)
