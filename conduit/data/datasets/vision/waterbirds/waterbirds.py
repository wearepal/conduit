from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union

import pandas as pd
from ranzen import parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import UrlFileInfo, download_from_url
from conduit.data.datasets.vision import CdtVisionDataset, ImageTform
from conduit.data.structures import TernarySample

__all__ = [
    "Waterbirds",
    "WaterbirdsSplit",
    "SampleType",
]


class WaterbirdsSplit(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


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
    _ORIGINAL_METADATA_FILENAME: ClassVar[str] = "metadata.csv"
    # Path to the 'fixed' metadata featured in 'MaskTune' and downloaded from the link provided in
    # the official repo of said paper:
    # 'https://drive.google.com/file/d/1xPNYQskEXuPhuqT5Hj4hXPeJa9jh7liL/view'.
    # To quote the paper:
    # "We discovered and fixed two problems with the Waterbirds dataset: a) Because the background
    # images in the Places dataset (Zhou et al., 2017) may already contain bird images, multiple
    # birds may appear in an image after overlaying the segmented bird images from the Caltech-UCSD
    # Birds- 200-2011 (CUB) dataset (Wah et al., 2011). For example, the label of an image may be
    # “landbird”, but the image contains both land and water birds. Such images were removed from
    # the dataset. b) Because the names of the species are similar, some land birds have been
    # mislabeled as waterbirds"
    _FIXED_METADATA_FILEPATH: ClassVar[Path] = Path(__file__).parent / "waterbirds_fixed.csv.zip"

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
        fixed: bool = False,
    ) -> None:
        self.split = WaterbirdsSplit[split.upper()] if isinstance(split, str) else split
        self.root = Path(root)
        self._base_dir = self.root / self._BASE_DIR_NAME
        self.fixed = fixed
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

        metadata_fp = (
            self._FIXED_METADATA_FILEPATH
            if self.fixed
            else self._base_dir / self._ORIGINAL_METADATA_FILENAME
        )
        # Read in metadata
        # Note: metadata is one-indexed.
        self.metadata = pd.read_csv(metadata_fp)
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
