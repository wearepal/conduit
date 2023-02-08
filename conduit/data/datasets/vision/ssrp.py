"""SSRP Dataset."""
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, Union, cast

import pandas as pd
from ranzen import parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import GdriveFileInfo, download_from_gdrive
from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["SSRP", "SSRPSplit"]


class SSRPSplit(Enum):
    TASK = "Task"
    PRETRAIN = "Pre_Train"


class SSRP(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    SampleType: TypeAlias = TernarySample
    Split: TypeAlias = SSRPSplit

    _FILE_INFO: ClassVar[GdriveFileInfo] = GdriveFileInfo(
        name="ghaziabad.zip", id="1RE4srtC63VnyU0e1qx16QNdjyyQXg2hj"
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Union[SSRPSplit, str] = SSRPSplit.PRETRAIN,
        download: bool = True,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.root = Path(root)
        self._base_dir = self.root / self.__class__.__name__
        self._metadata_path = self._base_dir / "metadata.csv"
        self.download = download
        self.split = SSRPSplit(split)

        if self.download:
            download_from_gdrive(file_info=self._FILE_INFO, root=self._base_dir, logger=self.logger)
        if not self._check_unzipped():
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )
        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self._base_dir / "metadata.csv")
        self.metadata = self.metadata.loc[self.metadata["split"].to_numpy() == self.split.value]

        x = self.metadata["filepath"].to_numpy()
        y = torch.as_tensor(self.metadata["class_le"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["season_le"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)

    def _check_unzipped(self) -> bool:
        return (self._base_dir / "Ghaziabad").is_dir()

    def _extract_metadata(self) -> None:
        """Extract concept/context/superclass information from the image filepaths and it save to csv."""
        self.logger.info("Extracting metadata.")
        image_paths: List[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            # Glob images from child folders recusrively, excluding hidden files
            image_paths.extend(self._base_dir.glob(f"**/[!.]*.{ext}"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str)
        metadata = cast(
            pd.DataFrame,
            filepaths.str.split("/", expand=True)
            .dropna(axis=1)
            .rename(columns={0: "region", 1: "split", 2: "class", 3: "filename"}),
        )
        # Extract the seasonal metadata from the filenames
        metadata["season"] = metadata["filename"].str.split(r"\((.*?)\s.*", expand=True)[1]
        metadata["filepath"] = filepaths
        metadata.sort_index(axis=1, inplace=True)
        metadata.sort_values(by=["filepath"], axis=0, inplace=True)
        metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path)

    @staticmethod
    def _label_encode_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
        """Label encode the extracted concept/context/superclass information."""
        for col in metadata.columns:
            # Skip over filepath and filename columns
            if "file" not in col:
                # Add a new column containing the label-encoded data
                metadata[f"{col}_le"] = metadata[col].factorize()[0]
        return metadata
