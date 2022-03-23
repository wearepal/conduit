"""CelebA Dataset."""
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import numpy as np
import pandas as pd
from ranzen import parsable, str_to_enum
from ranzen.decorators import enum_name_str
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import GdriveFileInfo, ImageTform, download_from_gdrive
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample

__all__ = ["CelebA", "CelebAttr", "CelebASplit"]


@enum_name_str
class CelebAttr(Enum):
    Five_o_Clock_Shadow = auto()
    Arched_Eyebrows = auto()
    Attractive = auto()
    Bags_Under_Eyes = auto()
    Bald = auto()
    Bangs = auto()
    Big_Lips = auto()
    Big_Nose = auto()
    Black_Hair = auto()
    Blond_Hair = auto()
    Blurry = auto()
    Brown_Hair = auto()
    Bushy_Eyebrows = auto()
    Chubby = auto()
    Double_Chin = auto()
    Eyeglasses = auto()
    Goatee = auto()
    Gray_Hair = auto()
    Heavy_Makeup = auto()
    High_Cheekbones = auto()
    Male = auto()
    Mouth_Slightly_Open = auto()
    Mustache = auto()
    Narrow_Eyes = auto()
    No_Beard = auto()
    Oval_Face = auto()
    Pale_Skin = auto()
    Pointy_Nose = auto()
    Receding_Hairline = auto()
    Rosy_Cheeks = auto()
    Sideburns = auto()
    Smiling = auto()
    Straight_Hair = auto()
    Wavy_Hair = auto()
    Wearing_Earrings = auto()
    Wearing_Hat = auto()
    Wearing_Lipstick = auto()
    Wearing_Necklace = auto()
    Wearing_Necktie = auto()
    Young = auto()


class CelebASplit(Enum):
    train = 0
    val = 1
    test = 2


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
        superclass: Union[CelebAttr, str] = CelebAttr.Smiling,
        subclass: Union[CelebAttr, str] = CelebAttr.Male,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[CelebASplit, str]] = None,
    ) -> None:

        self.superclass = str_to_enum(str_=superclass, enum=CelebAttr)
        self.subclass = str_to_enum(str_=subclass, enum=CelebAttr)
        self.split = str_to_enum(str_=split, enum=CelebASplit) if isinstance(split, str) else split

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
