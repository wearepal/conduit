"""Ecoacostics dataset provided A. Eldridge et al.
    Alice Eldridge, Paola Moscoso, Patrice Guyot, & Mika Peck. (2018).
    Data for "Sounding out Ecoacoustic Metrics: Avian species richness
    is predicted by acoustic indices in temperate but not tropical
    habitats" (Final) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.1255218
"""
from __future__ import annotations
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import ClassVar, Optional, Union
import zipfile

from bolts.data.datasets.audio.base import PBAudioDataset
from bolts.data.datasets.utils import AudioTform, FileInfo

__all__ = ["EcoacousticsDS"]


class EcoacousticsDS(PBAudioDataset):
    """Dataset for audio data collected in various geographic locations."""

    _FILE_INFO: ClassVar[FileInfo] = FileInfo(name="EcoacousticsDS.zip", id="PLACEHOLDER")
    _BASE_FOLDER: ClassVar[str] = "EcoacousticsDS"

    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[AudioTform] = None,
    ) -> None:
        self.root = Path(root)
        self.download = download
        self._base_dir = self.root / self._BASE_FOLDER
        self._acoustic_indices_dir = self._base_dir / "AvianID_AcousticIndices"

        # Acoustic indices files
        self._EC_AI_path = self._acoustic_indices_dir / "EC_AI.csv"
        self._UK_AI_path = self._acoustic_indices_dir / "UK_AI.csv"

        # Check data is downloaded and unzipped.
        if self.download:
            pass  # TODO: Implement download util function.
        elif not self._files_unzipped():
            raise RuntimeError(
                f"Data not found atlocation {self._base_dir.resolve()}. " "Have you downloaded it?"
            )

        if self._acoustic_indices_dir.exists():
            pass

    def _files_unzipped(self) -> bool:
        dir = self.root / self._BASE_FOLDER
        zip_checks = [
            zipfile.is_zipfile(join(dir, f)) for f in listdir(dir) if isfile(join(dir, f))
        ]
        return False if True in zip_checks else True
