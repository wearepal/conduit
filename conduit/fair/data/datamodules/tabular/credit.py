"""Credit Dataset."""
from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CreditDataModule", "CreditSens"]


class CreditSens(Enum):
    sex = "Sex"
    custom = "Custom"


@attr.define(kw_only=True)
class CreditDataModule(EthicMlDataModule):
    """Data Module for the Credit Dataset."""

    sens_feat: CreditSens = CreditSens.sex
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:
        return em.credit(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
