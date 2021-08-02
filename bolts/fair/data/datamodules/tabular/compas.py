"""COMPAS Dataset."""
from enum import Enum
from typing import Optional, Union

import ethicml as em
from ethicml.preprocessing.scaling import ScalerType
from kit import parsable
from kit.torch import TrainingMode

from .base import TabularDataModule

__all__ = ["CompasDataModule"]


class CompasSens(Enum):
    sex = "Sex"
    race = "Race"
    raceSex = "Race-Sex"


class CompasDataModule(TabularDataModule):
    """COMPAS Dataset."""

    @parsable
    def __init__(
        self,
        sens_feat: CompasSens = CompasSens.sex,
        disc_feats_only: bool = False,
        # Below are super vars. Not doing *args **kwargs due to this being parsable
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
        scaler: Optional[ScalerType] = None,
        training_mode: Union[TrainingMode, str] = "epoch",
    ) -> None:
        super().__init__(
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
            scaler=scaler,
            training_mode=training_mode,
        )
        self.sens_feat = sens_feat
        self.disc_feats_only = disc_feats_only

    @property
    def em_dataset(self) -> em.Dataset:
        return em.compas(split=self.sens_feat.value, discrete_only=self.disc_feats_only)
