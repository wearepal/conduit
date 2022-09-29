import attr
from ethicml.data import Dataset, German, GermanSplits

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["GermanDataModule", "GermanSplits"]


@attr.define(kw_only=True)
class GermanDataModule(EthicMlDataModule):
    """Data Module for the German Credit Dataset."""

    sens_feat: GermanSplits = GermanSplits.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return German(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
