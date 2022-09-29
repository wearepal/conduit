"""Admissions Dataset."""
import attr
from ethicml.data import Admissions, AdmissionsSplits, Dataset

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["AdmissionsDataModule", "AdmissionsSplits"]


@attr.define(kw_only=True)
class AdmissionsDataModule(EthicMlDataModule):
    """Data Module for the Admissions Dataset."""

    sens_feat: AdmissionsSplits = AdmissionsSplits.GENDER
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> Dataset:
        return Admissions(
            split=self.sens_feat, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
