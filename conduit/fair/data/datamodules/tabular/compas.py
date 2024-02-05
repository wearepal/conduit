"""COMPAS Dataset."""

from dataclasses import dataclass

from ethicml.data import Compas, Dataset
from ethicml.data import CompasSplits as CompasSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["CompasDataModule", "CompasSens"]


@dataclass(kw_only=True)
class CompasDataModule(EthicMlDataModule):
    """COMPAS Dataset."""

    sens_feat: CompasSens = CompasSens.SEX
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Compas(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
