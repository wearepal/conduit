"""CelebA data-module."""
from __future__ import annotations
from typing import Any, Optional, Union

import albumentations as A
from kit import implements, parsable
from kit.torch.data import TrainingMode
from pytorch_lightning import LightningDataModule

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import ImageTform
from conduit.data.datasets.vision.waterbirds import Waterbirds, WaterbirdsSplit
from conduit.data.structures import TrainValTestSplit

__all__ = ["WaterbirdsDataModule"]


class WaterbirdsDataModule(CdtVisionDataModule):
    """Data-module for the CelebA dataset."""

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
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        use_predefined_splits: bool = False,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = "epoch",
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

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Waterbirds(root=self.root, download=True)

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_train_transforms(self) -> A.Compose:
        # This transform conforsm with that described in https://arxiv.org/abs/2008.06775,
        # differing from the that described in the original paper by Sagawa et al. in the
        # respect that the images are not centre-cropped before resizing (which purportedly
        # makes the task easier).
        base_transforms = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
            ]
        )
        normalization = super()._default_train_transforms
        return A.Compose([base_transforms, normalization])

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_test_transforms(self) -> A.Compose:
        return self._default_train_transforms

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # Split the data according to the pre-defined split indices
        if self.use_predefined_splits:
            train_data, val_data, test_data = (
                Waterbirds(root=self.root, split=split) for split in WaterbirdsSplit
            )
        # Split the data randomly according to test- and val-prop
        else:
            all_data = Waterbirds(root=self.root, transform=None)
            val_data, test_data, train_data = all_data.random_split(
                props=(self.val_prop, self.test_prop)
            )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
