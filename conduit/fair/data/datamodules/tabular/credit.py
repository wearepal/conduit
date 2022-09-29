"""Credit Dataset."""
import attr
from ethicml.data import Credit, CreditSplits, Dataset

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CreditDataModule", "CreditSplits"]


@attr.define(kw_only=True)
class CreditDataModule(EthicMlDataModule):
    """Data Module for the Credit Dataset."""

    sens_feat: CreditSplits = CreditSplits.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Credit(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
