from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["GermanDataModule", "GermanSens"]


class GermanSens(Enum):
    sex = "Sex"
    custom = "Custom"


@attr.define(kw_only=True)
class GermanDataModule(EthicMlDataModule):
    """Data Module for the German Credit Dataset."""

    sens_feat: GermanSens = GermanSens.sex
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:
        return em.german(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
