"""Law Admissions Dataset."""
import attr
from ethicml.data import Dataset, Sqf
from ethicml.data import SqfSplits as SqfSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["SqfDataModule", "SqfSens"]


@attr.define(kw_only=True)
class SqfDataModule(EthicMlDataModule):
    """NYC Stop, Question, Frisk Dataset."""

    sens_feat: SqfSens = SqfSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Sqf(split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s)
