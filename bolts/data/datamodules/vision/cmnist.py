"""ColoredMNIST data-module."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import albumentations as A
from kit import implements, parsable
from kit.torch import TrainingMode, prop_random_split
import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data.dataset import ConcatDataset
from torchvision.datasets import MNIST

from bolts.data.datamodules import PBDataModule
from bolts.data.datasets.utils import ImageTform
from bolts.data.datasets.vision.cmnist import ColoredMNIST
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
        normalization = A.Normalize(self.norm_values)
        return A.Compose([base_transforms, normalization])

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        MNIST(root=str(self.root), download=True, train=True)
        MNIST(root=str(self.root), download=True, train=False)

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # TODO: Add more sophisticated (e.g. biased) splits
        train_data = ColoredMNIST(
            root=self.root,
            train=True,
            background=self.background,
            black=self.black,
            greyscale=self.greyscale,
            correlation=self.correlation,
            colors=self.colors,
            num_colors=self.num_colors,
            label_map=self.label_map,
        )
        test_data = ColoredMNIST(
            root=self.root,
            train=False,
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
            val_data, train_data_new = prop_random_split(dataset=train_data, props=self.val_prop)
            # Compute the channel-wise first and second moments
            channel_means, channel_stds = torch.std_mean(
                train_data.x[train_data_new.indices], dim=[0, 2, 3]
            )
            channel_means = np.mean(train_data.x[train_data_new.indices], axis=(0, 2, 3))
            channel_stds = np.std(train_data.x[train_data_new.indices], axis=(0, 2, 3))
        else:
            # Split the data randomly according to val- and test-prop
            all_data = ConcatDataset([train_data, test_data])
            all_x = np.concatenate([train_data.x, test_data.x], axis=0)
            val_data, test_data, train_data_new = prop_random_split(
                dataset=all_data, props=(self.val_prop, self.test_prop)
            )
            channel_means = np.mean(all_x[train_data_new.indices], axis=(0, 2, 3))
            channel_stds = np.std(all_x[train_data_new.indices], axis=(0, 2, 3))

        self.norm_values = MeanStd(mean=channel_means.tolist(), std=channel_stds.tolist())

        return TrainValTestSplit(train=train_data_new, val=val_data, test=test_data)
