"""COMPAS Dataset."""
import attr
from ethicml.data import Compas
from ethicml.data import CompasSplits as CompasSens
from ethicml.data import Dataset

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CompasDataModule", "CompasSens"]


@attr.define(kw_only=True)
class CompasDataModule(EthicMlDataModule):
    """COMPAS Dataset."""

    sens_feat: CompasSens = CompasSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Compas(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
