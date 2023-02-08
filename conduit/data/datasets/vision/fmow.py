from enum import auto
from pathlib import Path
from typing import ClassVar, Optional, Union

import pandas as pd
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import UrlFileInfo, download_from_url
from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["FMoW", "FMoWSplit", "FMoWSplitScheme"]


class FMoWSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    ID_VAL = "id_val"
    TEST = "test"
    ID_TEST = "id_test"


class FMoWSplitScheme(StrEnum):
    # Since 3 years are set aside for validation, the earliest year that can be used for splitting
    # -- such that the training set is not empty -- is 2003 + 3 = 2006.
    POST_2006 = auto()
    POST_2007 = auto()
    POST_2008 = auto()
    POST_2009 = auto()
    POST_2010 = auto()
    POST_2011 = auto()
    POST_2012 = auto()
    POST_2013 = auto()
    POST_2014 = auto()
    POST_2015 = auto()
    POST_2016 = auto()
    POST_2017 = auto()
    POST_2018 = auto()
    OFFICIAL = POST_2018
    "Official split: equivalent to 'POST_2018'"


SampleType: TypeAlias = TernarySample


class FMoW(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """The functional Map of the World (fMoW; land use / building classification) dataset.
    This is a port of the iWildCam dataset from `WILDS`_ which defines a preprocessing procedure for
    the dataset introduced in `fMoW`_.

    Per `fMoW`_: We present a new dataset, Functional Map of the World (fMoW), which aims to
    inspire the development of machine learning models capable of predicting the functional purpose
    of buildings and land use ['y' label] from temporal sequences of satellite images and a rich set of metadata
    features. The metadata provided with each image enables reasoning about location, time, sun
    angles, physical sizes, and other features when making predictions about objects in the image.
    Our dataset consists of over 1 million images from over 200 countries. For each image, we
    provide at least one bounding box annotation containing one of 63 categories, including a "false
    detection" category.

    For the sake of the WILDS becnhmark 'region' is defined as the grouping attribute on which the
    evaluation metrics are conditioned. This also holds here as we seek to keep everything
    functionally equivalent with respect to the WILDS version of the dataset.

    .. _WILDS:
        https://arxiv.org/abs/1711.07846

    .. _fMoW:
        https://arxiv.org/abs/1711.07846
    """

    SampleType: TypeAlias = TernarySample
    Split: TypeAlias = FMoWSplit
    SplitScheme: TypeAlias = FMoWSplitScheme

    _NUM_VAL_YEARS: ClassVar[int] = 3
    _BASE_DIR_NAME: ClassVar[str] = "fmow"
    _IMG_DIR_NAME: ClassVar[str] = "images"
    _METADATA_FILENAME: ClassVar[str] = "rgb_metadata.csv"
    _CC_MAPPER_FILENAME: ClassVar[str] = "country_code_mapping.csv"

    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="fmow.tar.gz",
        url="https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/",
        md5=None,
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[Union[FMoWSplit, str]] = None,
        split_scheme: Optional[Union[FMoWSplitScheme, str]] = FMoWSplitScheme.OFFICIAL,
        drop_other: bool = True,
        transform: Optional[ImageTform] = None,
        download: bool = True,
    ) -> None:
        """
        :param root: Root directory of the dataset.
        :param split: Which predefined split of the dataset to use. If ``None`` then the full
            (unsplit) dataset will be returned.
        :param transform: A function/transform that takes in a PIL or ndarray image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        :param split_scheme: Which splitting scheme to use. The task is treated as a time-series
            one: the data is split temporally such that the training data always precedes the test
            data in time. This is ignored if ``split`` is ``None`` (the entire dataset is used).
        :param drop_other: Whether to drop samples belonging to uncategorised ('Other') regions.
            Such samples make up a very small portion of the dataset and so we drop these by default
            given that they should be (according to WILDS) excluded from evaluation.
        :param download: If ``True``, downloads the dataset from the internet and puts it in the
            root directory. If the dataset is already downloaded, it is not downloaded again.
        :raises FileNotFoundError: If ``download=False`` and an existing dataset cannot be found in
            the root directory.
        """
        self.split = FMoWSplit(split) if isinstance(split, str) else split
        self.split_scheme = (
            FMoWSplitScheme(split_scheme) if isinstance(split_scheme, str) else split_scheme
        )
        self.root = Path(root)
        self._base_dir = self.root / self._BASE_DIR_NAME
        self._img_dir = self._base_dir / self._IMG_DIR_NAME
        self.download = download
        if not self._check_unzipped():
            if self.download:
                download_from_url(
                    file_info=self._FILE_INFO,
                    root=self.root,
                    logger=self.logger,
                    remove_finished=True,
                )

            else:
                raise FileNotFoundError(
                    f"Data not found at location {self._base_dir.resolve()}. Have you"
                    " downloaded it?"
                )

        # Read in metadata
        self.metadata = pd.read_csv(self._base_dir / self._METADATA_FILENAME)
        # filter out sequestered images
        self.metadata = self.metadata.loc[self.metadata['split'] != 'seq']
        country_codes = pd.read_csv(self._base_dir / self._CC_MAPPER_FILENAME)
        region_mapper = dict(zip(country_codes['alpha-3'], country_codes["region"]))
        # Map the country codes to regions using the supplementary file.
        self.metadata["region"] = self.metadata.country_code.map(region_mapper)
        # In the WILDS code the 'Other'  region is excluded from evaluation; since there are so few
        # samples sourced from 'Other' regions we instead take the tack of dropping these samples
        # by default.
        if drop_other:
            self.metadata.dropna(subset="region")
        else:
            self.metadata.fillna("Other", inplace=True)
        # label encode the class (land use category) labels -- we do this prior splitting to
        # ensure consistency in the mappings.
        self.metadata["category_le"] = self.metadata["category"].factorize()[0]
        # label encode the subgroup (reegion) labels -- we do this prior to splitting to ensure
        # consistency in the mappings.
        self.metadata["region_le"] = self.metadata["region"].factorize()[0]
        # extract the years from the timestamps.
        self.metadata["year"] = self.metadata.timestamp.str.split("-", expand=True)[0].astype(int)

        # Use a split of the data if one is specified, else just use all of the data
        if self.split is not None:
            split_mask = self.metadata.split == self.split.value
            self.metadata = self.metadata.loc[split_mask]

            if self.split_scheme is not None:
                test_threshold = int(self.split_scheme.value.split('_')[-1])
                # OOD test split
                if self.split is FMoWSplit.TEST:
                    scheme_mask = self.metadata.year >= test_threshold
                # OOD validation split
                elif self.split is FMoWSplit.VAL:
                    # # set aside the final years of the training set for validation
                    scheme_mask = (self.metadata.year < test_threshold) > (
                        self.metadata.year >= (test_threshold - self._NUM_VAL_YEARS)
                    )
                # ID (train/val/test) splits
                else:
                    scheme_mask = self.metadata.year < (test_threshold - self._NUM_VAL_YEARS)
                self.metadata = self.metadata.loc[scheme_mask]

        # Amend the filenames, which are actually based on the sample indexes.
        def index_to_fn(index: int) -> str:
            return f"rgb_img_{index}.png"

        self.metadata["img_filename"] = self.metadata.index.map(index_to_fn)
        x = self.metadata['img_filename'].to_numpy()
        y = torch.as_tensor(self.metadata["category_le"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["region_le"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._img_dir)

    def _check_unzipped(self) -> bool:
        return (self._img_dir).is_dir()
