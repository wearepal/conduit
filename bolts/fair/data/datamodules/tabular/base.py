"""This is where the inevitable common DataModule will live."""
from __future__ import annotations
from abc import abstractmethod

import ethicml as em
from ethicml.preprocessing.scaling import ScalerType
from kit import implements
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from bolts.fair.data.datamodules import BaseDataModule
from bolts.fair.data.datasets import DataTupleDataset

__all__ = ["TabularDataModule"]


class TabularDataModule(BaseDataModule):
    """COMPAS Dataset."""

    def __init__(
        self,
        val_split: float | int = 0.2,
        test_split: float | int = 0.2,
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 0,
        scaler: ScalerType | None = None,
        persist_workers: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        sample_with_replacement: bool = False,
    ):
        """COMPAS Dataset Module.

        Args:
            val_split: Percent (float) or number (int) of samples to use for the validation split
            test_split: Percent (float) or number (int) of samples to use for the test split
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            seed: RNG Seed
            scaler: SKLearn style data scaler. Fit to train, applied to val and test.
            persist_workers: Use persistent workers in dataloader?
            pin_memory: Should the memory be pinned?
            stratified_sampling: Use startified sampling?
            sample_with_replacement: If using stratified sampling, should the samples be unique?
        """
        super().__init__(
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            batch_size=batch_size,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            stratified_sampling=stratified_sampling,
            sample_with_replacement=sample_with_replacement,
        )
        self.scaler = scaler if scaler is not None else StandardScaler()

    @property
    @abstractmethod
    def em_dataset(self) -> em.Dataset:
        ...

    @staticmethod
    def _get_splits(train_len: int, val_split: int | float) -> list[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len -= val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * train_len)
            train_len -= val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")

        return splits

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        self.dims = (
            len(self.em_dataset.discrete_features) + len(self.em_dataset.continuous_features),
        )

    @implements(LightningDataModule)
    def setup(self, stage: str | None = None) -> None:
        self.datatuple = self.em_dataset.load(ordered=True)

        data_len = int(self.datatuple.x.shape[0])
        num_train_val, num_test = self._get_splits(data_len, self.test_prop)
        train_val, test = em.train_test_split(
            data=self.datatuple,
            train_percentage=(1 - (num_test / data_len)),
            random_seed=self.seed,
        )
        num_train, num_val = self._get_splits(num_train_val, self.val_prop)
        train, val = em.train_test_split(
            data=train_val,
            train_percentage=(1 - (num_val / num_train_val)),
            random_seed=self.seed,
        )

        train, self.scaler = em.scale_continuous(
            self.em_dataset, datatuple=train, scaler=self.scaler
        )
        val, _ = em.scale_continuous(self.em_dataset, datatuple=val, scaler=self.scaler, fit=False)
        test, _ = em.scale_continuous(
            self.em_dataset, datatuple=test, scaler=self.scaler, fit=False
        )

        self._train_data = DataTupleDataset(
            train,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        self._val_data = DataTupleDataset(
            val,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        self._test_data = DataTupleDataset(
            test,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

    @property
    def train_data(self) -> Dataset:
        return self._train_data

    @property
    def val_data(self) -> Dataset:
        return self._val_data

    @property
    def test_data(self) -> Dataset:
        return self._test_data
