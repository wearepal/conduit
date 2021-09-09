from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union, cast

from kit.misc import str_to_enum
import pandas as pd
import torch

from conduit.data.datasets.utils import ImageTform, UrlFileInfo, download_from_url
from conduit.data.datasets.vision.base import CdtVisionDataset

__all__ = ["WaterbirdsDataset", "WaterbirdsSplit"]


class WaterbirdsSplit(Enum):
    train = 0
    val = 1
    test = 2


class WaterbirdsDataset(CdtVisionDataset):
    """The Waterbirds dataset.
    The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
    """

    _BASE_FOLDER: ClassVar[str] = "Waterbirds"
    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="Waterbirds.tar.gz",
        url="https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
        md5=None,
    )

    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[WaterbirdsSplit, str]] = None,
    ) -> None:

        if isinstance(split, str):
            split = str_to_enum(str_=split, enum=WaterbirdsSplit)

        self.root = Path(root)
        self._base_dir = self.root / self._BASE_FOLDER
        self.download = download
        if self.download:
            download_from_url(
                file_info=self._FILE_INFO,
                root=self.root,
                logger=self.logger,
                remove_finished=True,
            )
        else:
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )

        # Read in metadata
        # Note: metadata_df is one-indexed.
        self.metadata = pd.read_csv(self._base_dir / 'metadata.csv')
        if split is not None:
            split_indices = self.metadata["split"] == split.value
            self.metadata = cast(pd.DataFrame, self.metadata[split_indices])

        # Extract filenames
        x = self.metadata['img_filename'].to_numpy()
        y = torch.as_tensor(self.metadata["y"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["place"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)
