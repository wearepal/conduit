"""Adult Income Dataset."""
from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["AdultDataModule", "AdultSens"]


class AdultSens(Enum):
    sex = "Sex"
    race = "Race"
    raceBinary = "Race-Binary"
    raceSex = "Race-Sex"
    custom = "Custom"
    nationality = "Nationality"
    education = "Education"


@attr.define(kw_only=True)
class AdultDataModule(EthicMlDataModule):
    """UCI Adult Income Dataset."""

    bin_nationality: bool = False
    sens_feat: AdultSens = AdultSens.sex
    bin_race: bool = False
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:

        return em.adult(
            split=self.sens_feat.value,
            binarize_nationality=self.bin_nationality,
            discrete_only=self.disc_feats_only,
            binarize_race=self.bin_race,
            invert_s=self.invert_s,
        )
