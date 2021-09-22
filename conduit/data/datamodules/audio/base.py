"""Base class for audio datasets."""
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from kit.torch import TrainingMode
from torch import nn

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.structures import ImageSize

__all__ = ["CdtAudioDataModule"]


class CdtAudioDataModule(CdtDataModule):
    _input_size: ImageSize

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
    ):
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
        self.root = root

    @property
    def size(self) -> ImageSize:
        if hasattr(self, "_input_size"):
            return self._input_size
        if hasattr(self, "_train_data"):
            self._input_size = ImageSize(*self._train_data[0].x.shape)  # type: ignore
            return self._input_size
        raise AttributeError("Input size unavailable because setup has not yet been called.")

    @property
    @abstractmethod
    def _base_augmentations(self) -> nn.Sequential:
        ...
