"""CelebA Dataset."""
from enum import auto
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import numpy as np
import pandas as pd
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import GdriveFileInfo, ImageTform, download_from_gdrive
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample

__all__ = ["CelebA", "CelebAttr", "CelebASplit"]


class CelebAttr(StrEnum):
    FIVE_O_CLOCK_SHADOW = "Five_o_Clock_Shadow"
    ARCHED_EYEBROWS = "Arched_Eyebrows"
    ATTRACTIVE = "Attractive"
    BAGS_UNDER_EYES = "Bags_Under_Eyes"
    BALD = "Bald"
    BANGS = "Bangs"
    BIG_LIPS = "Big_Lips"
    BIG_NOSE = "Big_Nose"
    BLACK_HAIR = "Black_Hair"
    BLOND_HAIR = "Blond_Hair"
    BLURRY = "Blurry"
    BROWN_HAIR = "Brown_Hair"
    BUSHY_EYEBROWS = "Bushy_Eyebrows"
    CHUBBY = "Chubby"
    DOUBLE_CHIN = "Double_Chin"
    EYEGLASSES = "Eyeglasses"
    GOATEE = "Goatee"
    GRAY_HAIR = "Gray_Hair"
    HEAVY_MAKEUP = "Heavy_Makeup"
    HIGH_CHEEKBONES = "High_Cheekbones"
    MALE = "Male"
    MOUTH_SLIGHTLY_OPEN = "Mouth_Slightly_Open"
    MUSTACHE = "Mustache"
    NARROW_EYES = "Narrow_Eyes"
    NO_BEARD = "No_Beard"
    OVAL_FACE = "Oval_Face"
    PALE_SKIN = "Pale_Skin"
    POINTY_NOSE = "Pointy_Nose"
    RECEDING_HAIRLINE = "Receding_Hairline"
    ROSY_CHEEKS = "Rosy_Cheeks"
    SIDEBURNS = "Sideburns"
    SMILING = "Smiling"
    STRAIGHT_HAIR = "Straight_Hair"
    WAVY_HAIR = "Wavy_Hair"
    WEARING_EARRINGS = "Wearing_Earrings"
    WEARING_HAT = "Wearing_Hat"
    WEARING_LIPSTICK = "Wearing_Lipstick"
    WEARING_NECKLACE = "Wearing_Necklace"
    WEARING_NECKTIE = "Wearing_Necktie"
    YOUNG = "Young"


class CelebASplit(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


SampleType: TypeAlias = TernarySample


class CelebA(CdtVisionDataset[SampleType, Tensor, Tensor]):
    """CelebA dataset."""

    # The data is downloaded to `download_dir` / `CelebA` / `_IMAGE_DIR`.
    _IMAGE_DIR: ClassVar[str] = "img_align_celeba"
    # Google drive IDs, MD5 hashes and filenames for the CelebA files.
    _FILE_LIST: ClassVar[List[GdriveFileInfo]] = [
        GdriveFileInfo(
            name="img_align_celeba.zip",
            id="1zmsC4yvw-e089uHXj5EdP0BSZ0AlDQRR",
            md5="00d2c5bc6d35e252742224ab0c1e8fcb",
        ),
        GdriveFileInfo(
            name="list_attr_celeba.txt",
            id="1gxmFoeEPgF9sT65Wpo85AnHl3zsQ4NvS",
            md5="75e246fa4810816ffd6ee81facbd244c",
        ),
        GdriveFileInfo(
            name="list_eval_partition.txt",
            id="1ih_VMokoI774ErNWrb26lDeWlanUBpnX",
            md5="d32c9cbf5e040fd4025c592c306e6668",
        ),
    ]

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        superclass: Union[CelebAttr, str] = CelebAttr.SMILING,
        subclass: Union[CelebAttr, str] = CelebAttr.MALE,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[CelebASplit, str]] = None,
    ) -> None:

        self.superclass = CelebAttr(superclass)
        self.subclass = CelebAttr(subclass)
        self.split = CelebASplit(split) if isinstance(split, str) else split

        self.root = Path(root)
        self._base_dir = self.root / self.__class__.__name__
        image_dir = self._base_dir / self._IMAGE_DIR

        if download:
            download_from_gdrive(file_info=self._FILE_LIST, root=self._base_dir, logger=self.logger)
        elif not self._check_unzipped():
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )

        if self.split is None:
            skiprows = None
        else:
            # splits: information about which samples belong to train, val or test
            splits = (
                pd.read_csv(
                    self._base_dir / "list_eval_partition.txt",
                    delim_whitespace=True,
                    names=["split"],
                )
                .to_numpy()
                .squeeze()
            )
            skiprows = (splits != self.split.value).nonzero()[0] + 2
        attrs = pd.read_csv(
            self._base_dir / "list_attr_celeba.txt",
            delim_whitespace=True,
            header=1,
            usecols=[self.superclass.name, self.subclass.name],
            skiprows=skiprows,
        )

        x = np.array(attrs.index)
        s_unmapped = torch.as_tensor(attrs[self.subclass.name].to_numpy())
        y_unmapped = torch.as_tensor(attrs[self.superclass.name].to_numpy())
        # map from {-1, 1} to {0, 1}
        s_binary = torch.div(s_unmapped + 1, 2, rounding_mode='floor')
        y_binary = torch.div(y_unmapped + 1, 2, rounding_mode='floor')

        super().__init__(x=x, y=y_binary, s=s_binary, transform=transform, image_dir=image_dir)

    def _check_unzipped(self) -> bool:
        return (self._base_dir / self._IMAGE_DIR).is_dir()
