"""COMPAS Dataset."""
from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CompasDataModule", "CompasSens"]


class CompasSens(Enum):
    sex = "Sex"
    race = "Race"
    raceSex = "Race-Sex"


@attr.define(kw_only=True)
class CompasDataModule(EthicMlDataModule):
    """COMPAS Dataset."""

    sens_feat: CompasSens = CompasSens.sex
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:
        return em.compas(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
