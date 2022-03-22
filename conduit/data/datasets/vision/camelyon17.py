from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Optional, Union, cast

import pandas as pd
from ranzen.decorators import enum_name_str, parsable
from ranzen.misc import str_to_enum
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import ImageTform, UrlFileInfo, download_from_url
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample

__all__ = [
    "Camelyon17",
    "Camelyon17Split",
    "Camelyon17SplitScheme",
]


@enum_name_str
class Camelyon17SplitScheme(Enum):
    official = auto()
    """Oficial split."""

    mixed_to_test = auto()
    """ 
    For the mixed-to-test setting, slide 23 (corresponding to patient 042, node 3 in the
    original dataset) is moved from the test set to the training set
    """


class Camelyon17Split(Enum):
    train = 0
    id_val = 1
    test = 2
    val = 3


@enum_name_str
class Camelyon17Attr(Enum):
    tumor = auto()
    center = auto()
    slide = auto()


SampleType: TypeAlias = TernarySample


class Camelyon17(CdtVisionDataset[SampleType, Tensor, Tensor]):
    """
    The CAMELYON17-WILDS histopathology dataset.
    This is a modified version of the original CAMELYON17 dataset.

    Supported `split_scheme`:
        - 'official'
        - 'mixed-to-test'

    Input (x):
        96x96 image patches extracted from histopathology slides.

    Label (y):
        y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.

    Metadata:
        Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
        and the slide it came from (integer from 0 to 49).

    Website:
        https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="camelyon17_v1.0.tar.gz",
        url="https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/",
        md5=None,
    )
    _TEST_CENTER: ClassVar[int] = 2
    _VAL_CENTER: ClassVar[int] = 1

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[Camelyon17Split, str]] = None,
        split_scheme: Union[Camelyon17SplitScheme, str] = Camelyon17SplitScheme.official,
        superclass: Union[Camelyon17Attr, str] = Camelyon17Attr.tumor,
        subclass: Union[Camelyon17Attr, str] = Camelyon17Attr.center,
    ) -> None:

        self.superclass = str_to_enum(str_=superclass, enum=Camelyon17Attr)
        self.subclass = str_to_enum(str_=subclass, enum=Camelyon17Attr)

        self.split = (
            str_to_enum(str_=split, enum=Camelyon17Split) if isinstance(split, str) else split
        )
        self.split_scheme = (
            str_to_enum(str_=split_scheme, enum=Camelyon17SplitScheme)
            if isinstance(split_scheme, str)
            else split_scheme
        )
        self.root = Path(root)
        self._base_dir = self.root / "camelyon17_v1.0"
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
        self.metadata = pd.read_csv(
            self._base_dir / 'metadata.csv', index_col=0, dtype={"patient": "str"}
        )
        if self.split_scheme is Camelyon17SplitScheme.mixed_to_test:
            # For the mixed-to-test setting,
            # we move slide 23 (corresponding to patient 042, node 3 in the original dataset)
            # from the test set to the training set
            slide_mask = self.metadata["slide"] == 23
            self.metadata.loc[slide_mask, "split"] = Camelyon17Split.train.value
        # Use an official split of the data, if 'split' is specified, else just use all
        # of the data
        val_center_mask = self.metadata["center"] == self._VAL_CENTER
        test_center_mask = self.metadata["center"] == self._TEST_CENTER
        self.metadata.loc[val_center_mask, "split"] = Camelyon17Split.val.value
        self.metadata.loc[test_center_mask, "split"] = Camelyon17Split.test.value

        if self.split is not None:
            split_indices = self.metadata["split"] == self.split.value
            self.metadata = cast(pd.DataFrame, self.metadata[split_indices])

        # Construct filepaths from metadata
        def build_fp(row: pd.DataFrame) -> str:
            return "patches/patient_{0}_node_{1}/patch_patient_{0}_node_{1}_x_{2}_y_{3}.png".format(
                *row
            )

        x = (
            self.metadata[["patient", "node", "x_coord", "y_coord"]]
            .apply(build_fp, axis=1)
            .to_numpy()
        )
        # Extract superclass labels
        y = torch.as_tensor(self.metadata[str(self.superclass)].to_numpy(), dtype=torch.long)
        # Extract subclass labels
        s = torch.as_tensor(self.metadata[str(self.subclass)].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)
