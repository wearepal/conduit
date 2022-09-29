import attr
from ethicml.data import Dataset, Health
from ethicml.data import HealthSplits as HealthSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["HealthDataModule", "HealthSens"]


@attr.define(kw_only=True)
class HealthDataModule(EthicMlDataModule):
    """Data Module for the Heritage Health Dataset."""

    sens_feat: HealthSens = HealthSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Health(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
