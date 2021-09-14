"""Ecoacostics dataset provided A. Eldridge et al.
    Alice Eldridge, Paola Moscoso, Patrice Guyot, & Mika Peck. (2018).
    Data for "Sounding out Ecoacoustic Metrics: Avian species richness
    is predicted by acoustic indices in temperate but not tropical
    habitats" (Final) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.1255218
"""
from __future__ import annotations
import math
from os import mkdir
from pathlib import Path
import shutil
import subprocess
from typing import ClassVar, Optional, Union
import zipfile

from kit import parsable
from kit.misc import str_to_enum
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from torchvision.datasets.utils import (
    _decompress,
    _detect_file_type,
    check_integrity,
    download_url,
)
from tqdm import tqdm
from typing_extensions import Literal

from conduit.data.datasets.audio.base import CdtAudioDataset
from conduit.data.datasets.utils import (
    AudioTform,
    GdriveFileInfo,
    UrlFileInfo,
    download_from_url,
)
from conduit.types import SoundscapeAttr

__all__ = ["Ecoacoustics"]


Extension = Literal[".pt", ".wav"]


class Ecoacoustics(CdtAudioDataset):
    """Dataset for audio data collected in various geographic locations."""

    INDICES_DIR: ClassVar[str] = "AvianID_AcousticIndices"
    METADATA_FILENAME: ClassVar[str] = "metadata.csv"

    _FILE_INFO: ClassVar[GdriveFileInfo] = GdriveFileInfo(name="Ecoacoustics.zip", id="PLACEHOLDER")
    _BASE_FOLDER: ClassVar[str] = "Ecoacoustics"
    _EC_LABELS_FILENAME: ClassVar[str] = "EC_AI.csv"
    _UK_LABELS_FILENAME: ClassVar[str] = "UK_AI.csv"
    _PROCESSED_DIR: ClassVar[str] = "processed_audio"
    _AUDIO_LEN: ClassVar[float] = 60.0  # Audio samples' durations in seconds.

    _INDICES_FILE_INFO: UrlFileInfo = UrlFileInfo(
        name="AvianID_AcousticIndices.zip",
        url="https://zenodo.org/record/1255218/files/AvianID_AcousticIndices.zip",
        md5="b23208eb7db3766a1d61364b75cb4def",
    )

    _AUDIO_FILE_INFO: list[UrlFileInfo] = [
        UrlFileInfo(
            name="EC_BIRD.zip",
            url="https://zenodo.org/record/1255218/files/EC_BIRD.zip",
            md5="d427e904af1565dbbfe76b05f24c258a",
        ),
        UrlFileInfo(
            name="UK_BIRD.zip",
            url="https://zenodo.org/record/1255218/files/UK_BIRD.zip",
            md5="e1e58b224bb8fb448d1858b9c9ee0d8c",
        ),
    ]

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        preprocessing_transform: Optional[AudioTform],
        transform: Optional[AudioTform],
        download: bool = True,
        target_attr: Union[SoundscapeAttr, str] = SoundscapeAttr.habitat,
        resample_rate: int = 22050,
        specgram_segment_len: float = 15,
    ) -> None:

        self.root = Path(root).expanduser()
        self.download = download
        self.base_dir = self.root / self._BASE_FOLDER
        self.labels_dir = self.base_dir / self.INDICES_DIR
        self._processed_audio_dir = self.base_dir / self._PROCESSED_DIR
        self._metadata_path = self.base_dir / self.METADATA_FILENAME
        self.ec_labels_path = self.labels_dir / self._EC_LABELS_FILENAME
        self.uk_labels_path = self.labels_dir / self._UK_LABELS_FILENAME

        if isinstance(target_attr, str):
            target_attr = str_to_enum(str_=target_attr, enum=SoundscapeAttr)
        self.target_attr = target_attr
        self.specgram_segment_len = specgram_segment_len
        self.resample_rate = resample_rate
        self._n_sgram_segments = int(self._AUDIO_LEN / specgram_segment_len)
        self.preprocessing_transform = preprocessing_transform

        if self.download:
            self._download_files()
        self._check_files()

        if not self._processed_audio_dir.exists():
            self._preprocess_audio()

        # Extract labels from indices files.
        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self.base_dir / self.METADATA_FILENAME)

        x = self.metadata["filePath_pt"].to_numpy()
        y = torch.as_tensor(self.metadata[f'{self.target_attr}_le'])
        s = None

        super().__init__(x=x, y=y, s=s, transform=transform, audio_dir=self.base_dir)

    def _check_integrity(self, file_info: UrlFileInfo) -> bool:
        fpath = self.base_dir / file_info.name
        if not check_integrity(str(fpath), file_info.md5):
            return False
        self.log(f"{file_info.name} already downloaded.")
        return True

    def _check_files(self) -> bool:
        """Check necessary files are present and unzipped."""

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Indices file not found at location {self.base_dir.resolve()}."
                "Have you downloaded it?"
            )
        if zipfile.is_zipfile(self.labels_dir):
            raise RuntimeError("Indices file not unzipped.")

        for dir_ in ["UK_BIRD", "EC_BIRD"]:
            path = self.base_dir / dir_
            if not path.exists():
                raise RuntimeError(
                    f"Data not found at location {self.base_dir.resolve()}. Have you downloaded it?"
                )
            if zipfile.is_zipfile(dir_):
                raise RuntimeError(f"{dir_} file not unzipped.")

        return True

    @staticmethod
    def _label_encode_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
        """Label encode the extracted concept/context/superclass information."""
        col_list = [str(col) for col in metadata.columns]
        for col in col_list:
            # Skip over filepath and filename columns
            if "file" not in col and "File" not in col:
                # Add a new column containing the label-encoded data
                metadata[f"{col}_le"] = metadata[col].factorize()[0]
        return metadata

    def _download_files(self) -> None:
        """Download all files necessary for dataset to work."""

        # Create necessary directories if they doesn't already exist.
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self._check_integrity(self._INDICES_FILE_INFO):
            download_from_url(file_info=self._INDICES_FILE_INFO, root=self.base_dir)
        for finfo in self._AUDIO_FILE_INFO:
            if not self._check_integrity(finfo):
                self.download_and_extract_archive_jar(finfo)

    def download_and_extract_archive_jar(
        self, finfo: UrlFileInfo, *, remove_finished: bool = False
    ) -> None:
        download_url(finfo.url, str(self.base_dir), finfo.name, finfo.md5)

        archive = self.base_dir / finfo.name
        self.log(f"Extracting {archive}")

        _, archive_type, _ = _detect_file_type(str(archive))
        if not archive_type:
            _ = _decompress(
                str(archive),
                str((self.base_dir / finfo.name).with_suffix("")),
                remove_finished=remove_finished,
            )
            return

        try:
            subprocess.run(["jar", "-xvf", str(archive)], check=True, cwd=self.base_dir)
        except subprocess.CalledProcessError:
            self.log(
                "Tried to extract malformed .zip file using Java."
                "However, there was a problem. Is Java in your system path?"
            )
        if (self.base_dir / "__MACOSX").exists():
            shutil.rmtree(self.base_dir / "__MACOSX")

    def _extract_metadata(self) -> None:
        """Extract information such as labels from relevant csv files, combining them along with
        information on processed files to produce a master file."""
        self.log("Extracting metadata.")

        def gen_files_df(ext: Extension) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "filePath": str(path.relative_to(self.base_dir)),
                        "fileName": path.name,
                        "baseFile": str(path.stem).split('=', maxsplit=1)[0],
                    }
                    for path in self.base_dir.glob(f"**/*{ext}")
                ]
            )

        ec_labels = pd.read_csv(self.ec_labels_path, encoding="ISO-8859-1")
        uk_labels = pd.read_csv(self.uk_labels_path, encoding="ISO-8859-1")

        sgram_seg_metadata = gen_files_df(".pt")

        # Merge labels and metadata files.
        metadata = gen_files_df(".wav")
        metadata = metadata.merge(pd.concat([uk_labels, ec_labels], ignore_index=True), how="left")

        metadata = sgram_seg_metadata.merge(
            metadata, how='left', on='baseFile', suffixes=('_pt', '_wav')
        )
        metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path)

    def _preprocess_audio(self) -> None:
        """
        Applies transformation to audio samples then segments transformed samples and stores
        them as processed files.
        """

        if not self._processed_audio_dir.exists():
            mkdir(self._processed_audio_dir)

        waveform_paths = list(self.base_dir.glob("**/*.wav"))

        for path in tqdm(waveform_paths, desc="Preprocessing"):
            waveform_filename = path.stem
            waveform, sr = torchaudio.load(path)  # type: ignore
            waveform = F.resample(waveform, orig_freq=sr, new_freq=self.resample_rate)
            audio_len = waveform.size(-1) / self.resample_rate
            frac_remainder, num_segments = math.modf(audio_len / self.specgram_segment_len)
            num_segments = int(num_segments)

            if frac_remainder >= 0.5:
                self.log(
                    f"Length of audio file '{path.resolve()}' is not integer-divisible by "
                    f"{self.specgram_segment_len}: terminally zero-padding the file along the "
                    f"time-axis to compensate."
                )
                padding = torch.zeros(
                    waveform.size(0),
                    int(
                        (self.specgram_segment_len - (frac_remainder * self.specgram_segment_len))
                        * self.resample_rate
                    ),
                )
                waveform = torch.cat((waveform, padding), dim=-1)
                num_segments += 1
            if 0 < frac_remainder < 0.5:
                self.log(
                    f"Length of audio file '{path.resolve()}' is not integer-divisible by "
                    f"{self.specgram_segment_len} and not of sufficient length to be padded "
                    f"(fractional remainder must be greater than 0.5): discarding terminal segment."
                )
                waveform = waveform[
                    :, : int(num_segments * self.specgram_segment_len * self.resample_rate)
                ]

            waveform_segments = waveform.chunk(chunks=num_segments, dim=-1)
            for i, segment in enumerate(waveform_segments):
                specgram = (
                    self.preprocessing_transform(segment)
                    if self.preprocessing_transform is not None
                    else segment
                )
                torch.save(specgram, f=self._processed_audio_dir / f"{waveform_filename}={i}.pt")
