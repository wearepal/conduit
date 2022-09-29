"""Crime Dataset."""
from enum import Enum

import attr
from ethicml.data import Crime, Dataset

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CrimeDataModule", "CrimeSens"]


class CrimeSens(Enum):
    raceBinary = "Race-Binary"


@attr.define(kw_only=True)
class CrimeDataModule(EthicMlDataModule):
    """Data Module for the Crime Dataset."""

    sens_feat: CrimeSens = CrimeSens.raceBinary
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Crime(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
