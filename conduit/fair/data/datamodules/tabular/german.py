import attr
from ethicml.data import Dataset, German
from ethicml.data import GermanSplits as GermanSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["GermanDataModule", "GermanSens"]


@attr.define(kw_only=True)
class GermanDataModule(EthicMlDataModule):
    """Data Module for the German Credit Dataset."""

    sens_feat: GermanSens = GermanSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return German(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
