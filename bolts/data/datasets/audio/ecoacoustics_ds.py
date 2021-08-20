"""Ecoacostics dataset provided A. Eldridge et al.
    Alice Eldridge, Paola Moscoso, Patrice Guyot, & Mika Peck. (2018).
    Data for "Sounding out Ecoacoustic Metrics: Avian species richness
    is predicted by acoustic indices in temperate but not tropical
    habitats" (Final) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.1255218
"""
from __future__ import annotations
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import ClassVar, Optional, Union
import zipfile

import numpy as np
import pandas as pd
import torch
from typing_extensions import Literal

from bolts.data.datasets.audio.base import PBAudioDataset
from bolts.data.datasets.utils import AudioTform, FileInfo

__all__ = ["EcoacousticsDS"]
SoundscapeAttr = Literal["habitat", "site"]


class EcoacousticsDS(PBAudioDataset):
    """Dataset for audio data collected in various geographic locations."""

    LABELS_DIR: ClassVar[str] = "AvianID_AcousticIndices"
    METADATA_FILENAME: ClassVar[str] = "metadata.csv"

    _FILE_INFO: ClassVar[FileInfo] = FileInfo(name="EcoacousticsDS.zip", id="PLACEHOLDER")
    _BASE_FOLDER: ClassVar[str] = "EcoacousticsDS"
    _EC_LABELS_FILENAME: ClassVar[str] = "EC_AI.csv"
    _UK_LABELS_FILENAME: ClassVar[str] = "UK_AI.csv"
@parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        target_attr: SoundscapeAttr = "habitat",
        transform: Optional[AudioTform] = None,
    ) -> None:

        self.root = Path(root)
        self.download = download
        self._base_dir = self.root / self._BASE_FOLDER
        self.labels_dir = self._base_dir / self.LABELS_DIR
        self._metadata_path = self._base_dir / "metadata.csv"
        self.ec_labels_path = self.labels_dir / self._EC_LABELS_FILENAME
        self.uk_labels_path = self.labels_dir / self._UK_LABELS_FILENAME

        # Check data is downloaded and unzipped.
        if self.download:
            pass  # TODO: Implement download util function.
        elif not self._files_unzipped():
            raise RuntimeError(
                f"Data not found at location {self._base_dir.resolve()}. " "Have you downloaded it?"
            )

        if not self._metadata_path.exists():
            self._extract_metadata(target_attr)

        self.metadata = pd.read_csv(self._base_dir / "metadata.csv")

        x = self.metadata["filePath"].to_numpy()
        y = torch.as_tensor(self.metadata[f'{target_attr}_le'])
        s = None

        super().__init__(x=x, y=y, s=s, transform=transform, audio_dir=self._base_dir)

    def _files_unzipped(self) -> bool:
        dir = self.root / self._BASE_FOLDER
        zip_checks = [
            zipfile.is_zipfile(join(dir, f)) for f in listdir(dir) if isfile(join(dir, f))
        ]
        return False if True in zip_checks else True

    def _label_encode_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Label encode the extracted concept/context/superclass information."""
        for col in metadata.columns:
            # Skip over filepath and filename columns
            if "file" not in col:
                # Add a new column containing the label-encoded data
                metadata[f"{col}_le"] = metadata[col].factorize()[0]
        return metadata

    def _extract_metadata(self, target_attr) -> None:
        self.log("Extracting metadata.")
        waveform_paths: list[Path] = []
        waveform_paths.extend(self._base_dir.glob("**/*.wav"))

        # Extract filepaths and names.
        waveform_paths_str = [str(wvfrm.relative_to(self._base_dir)) for wvfrm in waveform_paths]
        filepaths = pd.Series(waveform_paths_str)
        filenames = pd.Series([path_str.split(os.sep)[-1] for path_str in waveform_paths_str])

        # Extract labels.
        ec_labels = pd.read_csv(self.ec_labels_path, encoding="ISO-8859-1")
        uk_labels = pd.read_csv(self.uk_labels_path, encoding="ISO-8859-1")
        all_labels = pd.concat([ec_labels, uk_labels])

        target_attrs: list[SoundscapeAttr] = []

        for f_name in filenames:
            matched_row = all_labels.loc[all_labels['fileName'] == f_name]
            target_attrs.append(np.nan if matched_row.empty else matched_row.iloc[0][target_attr])
        target_attrs = pd.Series(target_attrs)

        metadata = pd.DataFrame(
            {
                "fileName": filenames,
                "filePath": filepaths,
                f"{target_attr}": target_attrs,
            }
        )
        metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path)
