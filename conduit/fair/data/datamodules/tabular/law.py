"""Law Admissions Dataset."""
from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["LawDataModule", "LawSens"]


class LawSens(Enum):
    sex = "Sex"
    race = "Race"
    sexRace = "Sex-Race"


@attr.define(kw_only=True)
class LawDataModule(EthicMlDataModule):
    """LSAC Law Admissions Dataset."""

    sens_feat: LawSens = LawSens.sex
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:

        return em.law(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
