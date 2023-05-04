"""Ecoacostics dataset provided A. Eldridge et al.
    Alice Eldridge, Paola Moscoso, Patrice Guyot, & Mika Peck. (2018).
    Data for "Sounding out Ecoacoustic Metrics: Avian species richness
    is predicted by acoustic indices in temperate but not tropical
    habitats" (Final) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.1255218
"""
from enum import auto
import math
from pathlib import Path
import shutil
from typing import ClassVar, Final, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
import torchaudio  # type: ignore
from tqdm import tqdm
from typing_extensions import TypeAlias, override

from conduit.data.datasets.audio.base import CdtAudioDataset
from conduit.data.datasets.utils import AudioTform, UrlFileInfo, download_from_url
from conduit.data.structures import TernarySample

__all__ = ["Ecoacoustics", "SoundscapeAttr"]


class SoundscapeAttr(StrEnum):
    HABITAT = auto()
    SITE = auto()
    TIME = auto()
    NN = "NN"
    N0 = "N0"


SampleType: TypeAlias = TernarySample


class Ecoacoustics(CdtAudioDataset[SampleType, Tensor, Tensor]):
    """Dataset for audio data collected in various geographic locations."""

    _INDICES_DIR: ClassVar[str] = "AvianID_AcousticIndices"
    _METADATA_FILENAME: ClassVar[str] = "metadata.csv"
    _PBAR_COL: ClassVar[str] = "#00FF00"

    _EC_LABELS_FILENAME: ClassVar[str] = "EC_AI.csv"
    _UK_LABELS_FILENAME: ClassVar[str] = "UK_AI.csv"

    _FILE_INFO: List[UrlFileInfo] = [
        UrlFileInfo(
            name="AvianID_AcousticIndices.zip",
            url="https://zenodo.org/record/1255218/files/AvianID_AcousticIndices.zip",
            md5="b23208eb7db3766a1d61364b75cb4def",
        ),
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

    num_frames_in_segment: int
    _MAX_AUDIO_LEN: Final[int] = 60

    @parsable
    def __init__(
        self,
        root: str,
        *,
        target_attrs: List[SoundscapeAttr],
        transform: Optional[AudioTform] = None,
        download: bool = True,
        segment_len: float = 15,
        sample_rate: int = 48_000,  # This is the value that is present in the dataset
    ) -> None:
        self.root = Path(root).expanduser()
        self.download = download
        self.base_dir = self.root / self.__class__.__name__
        self.labels_dir = self.base_dir / self._INDICES_DIR / self._INDICES_DIR
        self.sample_rate = sample_rate  # set prior to segment length
        self.segment_len = segment_len
        self._metadata_path = self.base_dir / self._METADATA_FILENAME
        self.ec_labels_path = self.labels_dir / self._EC_LABELS_FILENAME
        self.uk_labels_path = self.labels_dir / self._UK_LABELS_FILENAME
        # map provided attributes (as string or enum) to permitted attributes (as strings)
        self.target_attrs = [str(SoundscapeAttr(elem)) for elem in target_attrs]

        if self.download:
            self._download_files()
        self._check_files()

        # Extract labels from indices files.
        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self.base_dir / self._METADATA_FILENAME)

        x = self.metadata["filePath"].to_numpy()
        y = torch.as_tensor(
            self._label_encode(self.metadata[self.target_attrs], inplace=True).to_numpy()
        )

        super().__init__(x=x, y=y, transform=transform, audio_dir=self.base_dir)

    @property
    def segment_len(self) -> float:
        return self._segment_len

    @segment_len.setter
    def segment_len(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Segment length must be positive.")
        value = min(value, self._MAX_AUDIO_LEN)
        self._segment_len = value
        self.num_frames_in_segment = int(self.segment_len * self.sample_rate)

    def _check_files(self) -> None:
        """Check necessary files are present and unzipped."""

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Indices file not found at location {self.labels_dir.resolve()}. "
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

    @staticmethod
    def _label_encode(
        data: Union[pd.DataFrame, pd.Series], inplace: bool = True
    ) -> Union[pd.DataFrame, pd.Series]:
        """Label encode the extracted concept/context/superclass information."""
        data = data.copy(deep=not inplace)
        if isinstance(data, pd.Series):
            if is_object_dtype(data) or is_categorical_dtype(data):
                data.update(data.factorize()[0])  # type: ignore
                data = data.astype(np.int64)
        else:
            for col in data.columns:
                # Add a new column containing the label-encoded data
                if is_object_dtype(data[col]) or is_categorical_dtype(data[col]):
                    data[col] = data[col].factorize()[0]
        return data

    def _download_files(self) -> None:
        """Download all files necessary for dataset to work."""
        # Create necessary directories if they don't already exist.
        self.base_dir.mkdir(parents=True, exist_ok=True)

        for finfo in self._FILE_INFO:
            download_from_url(
                file_info=finfo, root=self.base_dir, logger=self.logger, remove_finished=True
            )
        if (macosx_dir := self.base_dir / "__MACOSX").exists():
            shutil.rmtree(macosx_dir)

    def _extract_metadata(self) -> None:
        """Extract information such as labels from relevant csv files, combining them along with
        information on processed files to produce a master file."""

        self.logger.info("Extracting metadata.")
        # Process the metadata for samples from Ecuador.
        ec_labels = pd.read_csv(self.ec_labels_path, encoding="ISO-8859-1")
        ec_labels["filePath"] = "EC_BIRD/" + ec_labels["fileName"]
        # Process the metadata for samples from the UK.
        uk_labels = pd.read_csv(self.uk_labels_path, encoding="ISO-8859-1")
        # Some waveforms use full names for location i.e. BALMER and KNEPP, change indices file to do the same.
        uk_labels.replace(regex={"BA-": "BALMER-", "KN-": "KNEPP-"}, inplace=True)
        uk_labels["filePath"] = "UK_BIRD/" + uk_labels["fileName"]

        metadata = pd.concat([uk_labels, ec_labels])
        metadata = metadata[[(self.base_dir / fp).exists() for fp in metadata["filePath"]]]
        # metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path, index=False)

    def _preprocess_files(self) -> None:
        """
        Segment the waveform files based on :py:attr:`segment_len` and cache the file-segments.
        """
        if self.segment_len is None:
            return
        processed_audio_dir = self.base_dir / f"segment_len={self.segment_len}"
        processed_audio_dir.mkdir(parents=True, exist_ok=True)

        waveform_paths = self.base_dir / self.metadata["filePath"]  # type: ignore
        segment_filenames: List[Tuple[str, str]] = []
        for path in tqdm(waveform_paths, desc="Preprocessing", colour=self._PBAR_COL):
            waveform_filename = path.stem
            waveform, sr = torchaudio.load(path)  # type: ignore
            audio_len = waveform.size(-1) / sr
            frac_remainder, num_segments = math.modf(audio_len / self.segment_len)
            num_segments = int(num_segments)

            if frac_remainder >= 0.5:
                self.logger.debug(
                    (
                        f"Length of audio file '{path.resolve()}' is not integer-divisible by "
                        f"{self.segment_len}: terminally zero-padding the file along the "
                        "time-axis to compensate."
                    ),
                )
                padding = torch.zeros(
                    waveform.size(0),
                    int((self.segment_len - (frac_remainder * self.segment_len)) * sr),
                )
                waveform = torch.cat((waveform, padding), dim=-1)
                num_segments += 1
            if 0 < frac_remainder < 0.5:
                self.logger.debug(
                    (
                        f"Length of audio file '{path.resolve()}' is not integer-divisible by"
                        f" {self.segment_len} and not of sufficient length to be padded (fractional"
                        " remainder must be greater than 0.5): discarding terminal segment."
                    ),
                )
                waveform = waveform[:, : int(num_segments * self.segment_len * sr)]

            waveform_segments = waveform.chunk(chunks=num_segments, dim=-1)
            for seg_idx, segment in enumerate(waveform_segments):
                segment_filename = f"{waveform_filename}_{seg_idx}.wav"
                segment_filepath = processed_audio_dir / segment_filename
                torchaudio.save(  # type: ignore
                    filepath=segment_filepath,
                    src=segment,
                    sample_rate=sr,
                )
                segment_filenames.append(
                    (waveform_filename, str(segment_filepath.relative_to(self.base_dir)))
                )

        pd.DataFrame(segment_filenames, columns=["fileName", "filePath"]).to_csv(
            processed_audio_dir / "filepaths.csv", index=False
        )

    @override
    def load_sample(self, index: int) -> Tensor:
        path = self.audio_dir / self.x[index]

        # get metadata first
        metadata = torchaudio.info(path)  # type: ignore

        # compute number of frames to take with the real sample rate
        num_frames_segment = int(
            self.num_frames_in_segment / self.sample_rate * metadata.sample_rate
        )

        # get random sub-sample
        high = max(1, metadata.num_frames - num_frames_segment)
        frame_offset = torch.randint(low=0, high=high, size=(1,))

        # load segment
        waveform, _ = torchaudio.load(  # type: ignore
            path, num_frames=num_frames_segment, frame_offset=frame_offset
        )

        # resample to correct sample rate
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=metadata.sample_rate,
            new_freq=self.sample_rate,
        )

        return waveform
