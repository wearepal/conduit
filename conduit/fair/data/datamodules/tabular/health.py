from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["HealthDataModule", "HealthSens"]


class HealthSens(Enum):
    sex = "Sex"


@attr.define(kw_only=True)
class HealthDataModule(EthicMlDataModule):
    """Data Module for the Heritage Health Dataset."""

    sens_feat: HealthSens = HealthSens.sex  # Currently the only allowed value
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:
        return em.health(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
