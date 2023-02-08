from pathlib import Path
from typing import ClassVar, Optional, Union

import pandas as pd
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import UrlFileInfo, download_from_url
from conduit.data.structures import SubgroupSample, TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["IWildCam", "IWildCamSplit", "IWildCamUnlabeled"]


class IWildCamSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    ID_VAL = "id_val"
    TEST = "test"
    ID_TEST = "id_test"


class IWildCam(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """The iWildCam2020-WILDS dataset.
    This is a port of the iWildCam dataset from `WILDS`_ which itself is a  modified version of the
    original iWildCam2020 competition dataset introduced in `iWildCam`_.

    Per `iWildCam`_: Camera traps enable the automatic collection of large quantities of image data.
    Biologists all over the world use camera traps to monitor animal populations. We have recently
    been making strides towards automatic species classification ['y' label] in camera trap images
    ['s' label]. However, as we try to expand the geographic scope of these models we are faced with
    an interesting question: how do we train models that perform well on new (unseen during
    training) camera trap locations? Can we leverage data from other modalities, such as citizen
    science data and remote sensing data? In order to tackle this problem, we have prepared a
    challenge where the training data and test data are from different cameras spread across the
    globe. For each camera, we provide a series of remote sensing imagery that is tied to the
    location of the camera. We also provide citizen science imagery from the set of species seen in
    our data. The challenge is to correctly classify species ('y') in the test camera traps.

    .. _WILDS:
        https://arxiv.org/abs/2012.07421

    .. _iWildCam:
        https://arxiv.org/abs/2004.10340
    """

    SampleType: TypeAlias = TernarySample
    Split: TypeAlias = IWildCamSplit

    _BASE_DIR_NAME: ClassVar[str] = "iwildcam_v2.0"
    _IMG_DIR_NAME: ClassVar[str] = "train"
    _METADATA_FILENAME: ClassVar[str] = "metadata.csv"

    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="iwildcam.tar.gz",
        url="https://worksheets.codalab.org/rest/bundles/0xff56ea50fbf64aabbc4d09b2e8d50e18/contents/blob/",
        md5=None,
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[IWildCamSplit, str]] = None,
    ) -> None:
        """
        :param root: Root directory of the dataset.
        :param split: Which predefined split of the dataset to use. If ``None`` then the full
            (unsplit) dataset will be returned.
        :param transform: A function/transform that takes in a PIL or ndarray image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        :param download: If ``True``, downloads the dataset from the internet and puts it in the
            root directory. If the dataset is already downloaded, it is not downloaded again.
        :raises FileNotFoundError: If ``download=False`` and an existing dataset cannot be found in
            the root directory.
        """
        self.split = IWildCamSplit(split) if isinstance(split, str) else split
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
        # Use an official split of the data, if specified, else just use all of the data
        if self.split is not None:
            split_indices = self.metadata["split"] == self.split.value
            self.metadata = self.metadata.loc[split_indices]

        # Extract filenames
        x = self.metadata['filename'].to_numpy()
        # Extract class (species) labels
        y = torch.as_tensor(self.metadata["y"].to_numpy(), dtype=torch.long)
        # Extract camera-trap-location labels
        s = torch.as_tensor(self.metadata["location_remapped"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._img_dir)

    def _check_unzipped(self) -> bool:
        return (self._img_dir).is_dir()


class IWildCamUnlabeled(CdtVisionDataset[SubgroupSample, None, Tensor]):
    """An extra unlablled (with respect to 'y') split of the WILDS, introduced in `Ext_WILDS`_.

    `_Ext_WILDS`_ extends the WILDS benchmark to include curated unlabeled data that would be
    realistically obtainable in deployment. In the case of the iWildCam dataset specifically, the
    unlabeled data, comprising 819,120 images, is obtained from a set of WCS camera traps entirely
    disjoint with the labeled dataset, representative of unlabeled data from a newly-deployed sensor
    network.

    .. _Ext_WILDS:
        https://arxiv.org/abs/2012.07421
    """

    SampleType: TypeAlias = SubgroupSample
    Split: TypeAlias = IWildCamSplit

    _BASE_DIR_NAME: ClassVar[str] = "iwildcam_unlabeled_v1.0"
    _IMG_DIR_NAME: ClassVar[str] = "images"
    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="iwildcam_unlabeled.tar.gz",
        url="https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/",
        md5=None,
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.root = Path(root)
        self._base_dir = self.root / self._BASE_DIR_NAME
        self._img_dir = self._base_dir / self._IMG_DIR_NAME
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
        self.metadata = pd.read_csv(self._base_dir / 'metadata.csv')
        self.metadata["filename"] = self.metadata["uid"] + ".jpg"

        # Extract filenames
        x = self.metadata['filename'].to_numpy()
        # Extract camera-trap-location labels
        s = torch.as_tensor(self.metadata["location_remapped"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=None, s=s, transform=transform, image_dir=self._img_dir)

    def _check_unzipped(self) -> bool:
        return (self._img_dir).is_dir()
