"""Admissions Dataset."""
from enum import Enum

import attr
import ethicml as em

from conduit.fair.data.datamodules.tabular.base import EthicMlDataModule

__all__ = ["AdmissionsDataModule", "AdmissionsSens"]


class AdmissionsSens(Enum):
    gender = "Gender"


@attr.define(kw_only=True)
class AdmissionsDataModule(EthicMlDataModule):
    """Data Module for the Admissions Dataset."""

    sens_feat: AdmissionsSens = AdmissionsSens.gender
    disc_feats_only: bool = False

    @property
    def em_dataset(self) -> em.Dataset:
        return em.admissions(
            split=self.sens_feat.value, discrete_only=self.disc_feats_only, invert_s=self.invert_s
        )
