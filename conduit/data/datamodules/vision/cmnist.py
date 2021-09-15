"""ColoredMNIST data-module."""
from __future__ import annotations
from functools import partial
from typing import Any, Dict, List, Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import attr
from kit import implements
import numpy as np
from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.vision.cmnist import ColoredMNIST, ColoredMNISTSplit
from conduit.data.structures import MeanStd, TrainValTestSplit

__all__ = ["ColoredMNISTDataModule"]


@attr.define(kw_only=True)
class ColoredMNISTDataModule(CdtVisionDataModule):
    """Data-module for the ColoredMNIST dataset."""

    image_size: int = 32
    use_predefined_splits: bool = False
    # Colorization settings
    label_map: Optional[Dict[str, int]] = None
    colors: Optional[List[int]] = None
    num_colors: int = 10
    scale: float = 0.2
    correlation: float = 1.0
    binarize: bool = False
    greyscale: bool = False
    background: bool = False
    black: bool = True

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_train_transforms(self) -> A.Compose:
        base_transforms = A.Compose([A.Resize(self.image_size, self.image_size)])
        normalization = A.Compose([A.Normalize(self.norm_values), ToTensorV2])
        return A.Compose([base_transforms, normalization])

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_test_transforms(self) -> A.Compose:
        return self._default_train_transforms

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        MNIST(root=str(self.root), download=True, train=True)
        MNIST(root=str(self.root), download=True, train=False)

    @implements(CdtDataModule)
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
            val_data, train_data = train_data.random_split(props=self.val_prop)
        else:
            # Split the data randomly according to val- and test-prop
            train_data = fact_func(split=None)
            val_data, test_data, train_data = train_data.random_split(
                props=(self.val_prop, self.test_prop)
            )
        # Compute the channel-wise first and second moments
        channel_means = np.mean(train_data.x, axis=(0, 1, 2)) / 255.0
        channel_stds = np.std(train_data.x, axis=(0, 1, 2)) / 255.0

        self.norm_values = MeanStd(mean=channel_means.tolist(), std=channel_stds.tolist())

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
