from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union, cast

import pandas as pd
from ranzen import parsable, str_to_enum
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import ImageTform, UrlFileInfo, download_from_url
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample

__all__ = ["Waterbirds", "WaterbirdsSplit"]


class WaterbirdsSplit(Enum):
    train = 0
    val = 1
    test = 2


SampleType: TypeAlias = TernarySample


class Waterbirds(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """The Waterbirds dataset.

    The dataset was constructed by cropping out birds from the Caltech-UCSD Birds-200-2011 (CUB) dataset
    and transferring them onto backgrounds from the Places dataset.

    The train-test split is the same as the one used for the CUB dataset, with 20% of the training data chosen to
    serve as a validation set. For the validation and test sets, landbirds and waterbirds are equally
    distributed across land and water backgrounds (i.e., there are the same number of landbirds on land
    vs. water backgrounds, and separately, the same number of waterbirds on land vs. water backgrounds).
    This allows us to more accurately measure the performance of the rare groups, and it is particularly
    important for the Waterbirds dataset because of its relatively small size; otherwise, the smaller groups
    (waterbirds on land and landbirds on water) would have too few samples to accurately estimate performance on.
    """

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

        self.split = (
            str_to_enum(str_=split, enum=WaterbirdsSplit) if isinstance(split, str) else split
        )
        self.root = Path(root)
        self._base_dir = self.root / self.__class__.__name__
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
        # Note: metadata is one-indexed.
        self.metadata = pd.read_csv(self._base_dir / 'metadata.csv')
        # Use an official split of the data, if specified, else just use all
        # of the data
        if self.split is not None:
            split_indices = self.metadata["split"] == self.split.value
            self.metadata = cast(pd.DataFrame, self.metadata[split_indices])

        # Extract filenames
        x = self.metadata['img_filename'].to_numpy()
        # Extract class (land- vs. water-bird) labels
        y = torch.as_tensor(self.metadata["y"].to_numpy(), dtype=torch.long)
        # Extract place (land vs. water) labels
        s = torch.as_tensor(self.metadata["place"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)
