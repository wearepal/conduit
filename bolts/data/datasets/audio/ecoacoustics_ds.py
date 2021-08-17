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
from typing import ClassVar, Optional, Union, cast
import zipfile

import pandas as pd

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
        self._metadata_path = self._base_dir / "metadata.csv"

        # Check data is downloaded and unzipped.
        if self.download:
            pass  # TODO: Implement download util function.
        elif not self._files_unzipped():
            raise RuntimeError(
                f"Data not found at location {self._base_dir.resolve()}. " "Have you downloaded it?"
            )

        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self._base_dir / "metadata.csv")

        x = self.metadata["filepath"].to_numpy()
        y = None
        s = None

        super().__init__(x=x, y=y, s=s, transform=transform, audio_dir=self._base_dir)

    def _files_unzipped(self) -> bool:
        dir = self.root / self._BASE_FOLDER
        zip_checks = [
            zipfile.is_zipfile(join(dir, f)) for f in listdir(dir) if isfile(join(dir, f))
        ]
        return False if True in zip_checks else True

    def _extract_metadata(self) -> None:
        self.log("Extracting metadata.")
        waveform_paths: list[Path] = []
        waveform_paths.extend(self._base_dir.glob("**/*.wav"))

        waveform_paths_str = [str(wvfrm.relative_to(self._base_dir)) for wvfrm in waveform_paths]
        filepaths = pd.Series(waveform_paths_str)
        metadata = cast(
            pd.DataFrame,
            filepaths.str.split("/", expand=True).rename(  # type: ignore[attr-defined]
                columns={0: "filename"}
            ),
        )
        metadata['filepath'] = filepaths
        metadata.sort_values(by=["filepath"], axis=0, inplace=True)
        metadata.to_csv(self._metadata_path)
