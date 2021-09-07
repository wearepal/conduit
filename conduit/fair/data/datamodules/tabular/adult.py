"""Adult Income Dataset."""
from enum import Enum
from typing import Optional, Union

import ethicml as em
from ethicml.preprocessing.scaling import ScalerType
from kit import parsable
from kit.torch import TrainingMode

__all__ = ["AdultDataModule"]

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule


class AdultSens(Enum):
    sex = "Sex"
    race = "Race"
    raceBinary = "Race-Binary"
    raceSex = "Race-Sex"
    custom = "Custom"
    nationality = "Nationality"
    education = "Education"


class AdultDataModule(EthicMlDataModule):
    """UCI Adult Income Dataset."""

    @parsable
    def __init__(
        self,
        bin_nationality: bool = False,
        sens_feat: AdultSens = AdultSens.sex,
        bin_race: bool = False,
        discrete_feats_only: bool = False,
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
        self.bin_nat = bin_nationality
        self.sens_feat = sens_feat
        self.bin_race = bin_race
        self.disc_only = discrete_feats_only

    @property
    def em_dataset(self) -> em.Dataset:

        return em.adult(
            split=self.sens_feat.value,
            binarize_nationality=self.bin_nat,
            discrete_only=self.disc_only,
            binarize_race=self.bin_race,
        )
