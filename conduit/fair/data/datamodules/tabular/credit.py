"""Credit Dataset."""

from dataclasses import dataclass

from ethicml.data import Credit, Dataset
from ethicml.data import CreditSplits as CreditSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CreditDataModule", "CreditSens"]


@dataclass(kw_only=True)
class CreditDataModule(EthicMlDataModule):
    """Data Module for the Credit Dataset."""

    sens_feat: CreditSens = CreditSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Credit(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
