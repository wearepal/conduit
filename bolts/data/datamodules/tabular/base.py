"""Base class for tabular data-modules."""
from __future__ import annotations
from enum import Enum
from typing import Optional, Union

from kit import implements
from kit.torch import TrainingMode

from bolts import transforms as PBT
from bolts.data.datamodules import PBDataModule
from bolts.data.datasets.utils import extract_base_dataset
from bolts.data.datasets.wrappers import InstanceWeightedDataset, TabularTransformer
from bolts.structures import Stage

__all__ = ["TabularDataModule", "TabularNormalizer"]


class TabularNormalizer(Enum):
    minmax = PBT.MinMaxNormalization
    quantile = PBT.QuantileNormalization
    zscore = PBT.ZScoreNormalization


class TabularDataModule(PBDataModule):
    """Base data-module for tabular datasets."""

    _num_features: int

    def __init__(
        self,
        *,
        train_batch_size: int = 100,
        eval_batch_size: Optional[int] = 256,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 0,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        training_mode: Union[TrainingMode, str] = "epoch",
        feature_normalizer: Optional[TabularNormalizer] = TabularNormalizer.zscore,
        target_normalizer: Optional[TabularNormalizer] = None,
    ) -> None:
        """Base data-module for tabular data.

        Args:
            val_prop: Proprtion (float)  of samples to use for the validation split
            test_prop: Proportion (float) of samples to use for the test split
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            seed: RNG Seed
            scaler: SKLearn style data scaler. Fit to train, applied to val and test.
            persist_workers: Use persistent workers in dataloader?
            pin_memory: Should the memory be pinned?
            stratified_sampling: Use startified sampling?
        """
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
        self.feature_normalizer = (
            None if feature_normalizer is None else feature_normalizer.value(inplace=False)
        )
        self.target_normalizer = (
            None if feature_normalizer is None else target_normalizer.value(inplace=False)
        )

    @property
    def size(self) -> int:
        if hasattr(self, "_num_features"):
            return self._num_features
        if hasattr(self, "_train_data"):
            self._num_features = self._train_data[0].size(0)  # type: ignore
            return self._num_features
        raise AttributeError("size unavailable because setup has not yet been called.")

    @implements(PBDataModule)
    def setup(self, stage: Stage | None = None) -> None:
        train, val, test = self._get_splits()
        base_train, ss_indices = extract_base_dataset(train, return_subset_indices=True)
        self.feature_normalizer.fit(base_train.x[ss_indices])
        train = TabularTransformer(
            train, transform=self.feature_normalizer, target_transform=self.target_normalizer
        )
        if self.instance_weighting:
            train = InstanceWeightedDataset(train)
        self._train_data = train
        self._val_data = TabularTransformer(
            val, transform=self.feature_normalizer, target_transform=self.target_normalizer
        )
        self._test_data = TabularTransformer(
            test, transform=self.feature_normalizer, target_transform=self.target_normalizer
        )
