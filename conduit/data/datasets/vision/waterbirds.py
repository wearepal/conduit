from enum import auto
from pathlib import Path
from typing import ClassVar, Optional, Union

import pandas as pd  # type: ignore
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import ImageTform, UrlFileInfo, download_from_url
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample

__all__ = ["Waterbirds", "WaterbirdsSplit"]


class WaterbirdsSplit(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


SampleType: TypeAlias = TernarySample


class Waterbirds(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """The Waterbirds dataset introduced in `GDRO`_.

    The dataset was constructed by cropping out birds from the Caltech-UCSD Birds-200-2011 (CUB) dataset
    and transferring them onto backgrounds from the Places dataset.

    The train-test split is the same as the one used for the CUB dataset, with 20% of the training data chosen to
    serve as a validation set. For the validation and test sets, landbirds and waterbirds are equally
    distributed across land and water backgrounds (i.e., there are the same number of landbirds on land
    vs. water backgrounds, and separately, the same number of waterbirds on land vs. water backgrounds).
    This allows us to more accurately measure the performance of the rare groups, and it is particularly
    important for the Waterbirds dataset because of its relatively small size; otherwise, the smaller groups
    (waterbirds on land and landbirds on water) would have too few samples to accurately estimate performance on.

    .. _GDRO:
        Generalization<https://arxiv.org/abs/1911.08731>`__
    """

    SampleType: TypeAlias = TernarySample
    Split: TypeAlias = WaterbirdsSplit

    _BASE_DIR_NAME: ClassVar[str] = "Waterbirds"
    _METADATA_FILENAME: ClassVar[str] = "metadata.csv"

    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="Waterbirds.tar.gz",
        url="https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
        md5=None,
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[WaterbirdsSplit, str]] = None,
    ) -> None:

        self.split = WaterbirdsSplit(split) if isinstance(split, str) else split
        self.root = Path(root)
        self._base_dir = self.root / self._BASE_DIR_NAME
        self.download = download
        if self.download:
            download_from_url(
                file_info=self._FILE_INFO,
                root=self.root,
                logger=self.logger,
                remove_finished=True,
            )
        elif not self._check_unzipped():
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )

        # Read in metadata
        # Note: metadata is one-indexed.
        self.metadata = pd.read_csv(self._base_dir / self._METADATA_FILENAME)
        # Use an official split of the data, if specified, else just use all
        # of the data
        if self.split is not None:
            split_indices = self.metadata["split"] == self.split.value
            self.metadata = self.metadata.loc[split_indices]

        # Extract filenames
        x = self.metadata['img_filename'].to_numpy()
        # Extract class (land- vs. water-bird) labels
        y = torch.as_tensor(self.metadata["y"].to_numpy(), dtype=torch.long)
        # Extract place (land vs. water) labels
        s = torch.as_tensor(self.metadata["place"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)

    def _check_unzipped(self) -> bool:
        return (self._base_dir).is_dir()
