"""Adult Income Dataset."""

from dataclasses import dataclass

from ethicml.data import Adult, Dataset
from ethicml.data import AdultSplits as AdultSens

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["AdultDataModule", "AdultSens"]


@dataclass(kw_only=True)
class AdultDataModule(EthicMlDataModule):
    """UCI Adult Income Dataset."""

    bin_nationality: bool = False
    sens_feat: AdultSens = AdultSens.SEX
    bin_race: bool = False
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Adult(
            split=self.sens_feat,
            binarize_nationality=self.bin_nationality,
            discrete_only=self.disc_feats_only,
            binarize_race=self.bin_race,
            invert_s=self.invert_s,
        )
