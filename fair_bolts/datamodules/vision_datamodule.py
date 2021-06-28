"""Common components for an EthicML vision datamodule."""
import os
from typing import Optional, Union

from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule


class VisionBaseDataModule(BaseDataModule):
    """Base DataModule for this project."""

    def __init__(
        self,
        data_dir: Optional[str],
        batch_size: int,
        num_workers: int,
        val_split: Union[float, int],
        test_split: Union[float, int],
        y_dim: int,
        s_dim: int,
        seed: int,
        persist_workers: bool,
        pin_memory: bool,
        stratified_sampling: bool,
        sample_with_replacement: bool,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            stratified_sampling=stratified_sampling,
            sample_with_replacement=sample_with_replacement,
        )
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.seed = seed

        self._train_data: Optional[Dataset] = None
        self._test_data: Optional[Dataset] = None
        self._val_data: Optional[Dataset] = None

    @property
    def train_data(self) -> Dataset:
        return self._train_data

    @property
    def val_data(self) -> Dataset:
        return self._val_data

    @property
    def test_data(self) -> Dataset:
        return self._test_data
