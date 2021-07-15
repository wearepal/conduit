"""ColoredMNIST data-module."""
from typing import Any, Dict, List, Optional

import albumentations as A
from kit import implements, parsable
from kit.torch import prop_random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import ConcatDataset
from torchvision.datasets import MNIST

from bolts.data.datamodules import PBDataModule, TrainingMode
from bolts.data.datasets.vision.cmnist import ColoredMNIST
from bolts.data.structures import TrainValTestSplit

from .base import PBVisionDataModule

__all__ = ["ColoredMNISTDataModule"]


class ColoredMNISTDataModule(PBVisionDataModule):
    """Data-module for the ColoredMNIST dataset."""

    @parsable
    def __init__(
        self,
        root: str,
        image_size: int = 32,
        batch_size: int = 100,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: TrainingMode = TrainingMode.epoch,
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
    def _base_augmentations(self) -> A.Compose:
        return A.Compose([A.Resize(self.image_size, self.image_size)])

    @property  # type: ignore[misc]
    @implements(PBVisionDataModule)
    def _train_augmentations(self) -> A.Compose:
        return A.Compose([])

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        MNIST(root=str(self.root), download=True, train=True)
        MNIST(root=str(self.root), download=True, train=False)

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # TODO: Add more sophisticated (e.g. biased) splits
        train_data = ColoredMNIST(root=self.root, train=True)
        test_data = ColoredMNIST(root=self.root, train=False)
        # Use the predefined train and test splits, sampling the val split
        # randomly from the train split
        if self.use_predefined_splits:
            val_data, train_data = prop_random_split(dataset=train_data, props=self.val_prop)
        else:
            # Split the data randomly according to val- and test-prop
            all_data = ConcatDataset([train_data, test_data])
            val_data, test_data, train_data = prop_random_split(
                dataset=all_data, props=(self.val_prop, self.test_prop)
            )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
