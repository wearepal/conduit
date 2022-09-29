"""Law Admissions Dataset."""
import attr
from ethicml.data import Dataset, Law
from ethicml.data import LawSplits as LawSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["LawDataModule", "LawSens"]


@attr.define(kw_only=True)
class LawDataModule(EthicMlDataModule):
    """LSAC Law Admissions Dataset."""

    sens_feat: LawSens = LawSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Law(split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s)
