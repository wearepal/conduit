"""Tabular data-module."""
from __future__ import annotations
from abc import abstractmethod

import ethicml as em
from ethicml.preprocessing.scaling import ScalerType
from kit import implements
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler

from bolts.data.datamodules.base import PBDataModule
from bolts.data.structures import TrainValTestSplit
from bolts.fair.data.datasets import DataTupleDataset

__all__ = ["TabularDataModule"]


class TabularDataModule(PBDataModule):
    """Base data-module for tabular datasets."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        seed: int = 0,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        instance_weighting: bool = False,
        scaler: ScalerType | None = None,
    ):
        """COMPAS Dataset Module.

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
            batch_size=batch_size,
            num_workers=num_workers,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            seed=seed,
            test_prop=test_prop,
            val_prop=val_prop,
            stratified_sampling=stratified_sampling,
            instance_weighting=instance_weighting,
        )
        self.scaler = scaler if scaler is not None else StandardScaler()

    @property
    @abstractmethod
    def em_dataset(self) -> em.Dataset:
        ...

    @staticmethod
    def _get_split_sizes(train_len: int, val_prop: int | float) -> list[int]:
        """Computes split sizes for train and validation sets."""
        if isinstance(val_prop, int):
            train_len -= val_prop
            splits = [train_len, val_prop]
        elif isinstance(val_prop, float):
            val_len = int(val_prop * train_len)
            train_len -= val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_prop)}")

        return splits

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        self.dims = (
            len(self.em_dataset.discrete_features) + len(self.em_dataset.continuous_features),
        )

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        self.datatuple = self.em_dataset.load(ordered=True)

        data_len = int(self.datatuple.x.shape[0])
        num_train_val, num_test = self._get_split_sizes(data_len, self.test_prop)
        train_val, test = em.train_test_split(
            data=self.datatuple,
            train_percentage=(1 - (num_test / data_len)),
            random_seed=self.seed,
        )
        _, num_val = self._get_split_sizes(num_train_val, self.val_prop)
        train, val = em.train_test_split(
            data=train_val,
            train_percentage=(1 - (num_val / num_train_val)),
            random_seed=self.seed,
        )

        train, self.scaler = em.scale_continuous(
            self.em_dataset, datatuple=train, scaler=self.scaler  # type: ignore
        )
        val, _ = em.scale_continuous(self.em_dataset, datatuple=val, scaler=self.scaler, fit=False)
        test, _ = em.scale_continuous(
            self.em_dataset, datatuple=test, scaler=self.scaler, fit=False
        )

        train = DataTupleDataset(
            train,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        val = DataTupleDataset(
            val,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        test = DataTupleDataset(
            test,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )
        return TrainValTestSplit(train=train, val=val, test=test)
