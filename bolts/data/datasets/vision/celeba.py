"""CelebA Dataset."""
from __future__ import annotations
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Optional, Union

from kit import parsable
import numpy as np
import pandas as pd
import torch

from bolts.common import str_to_enum
from bolts.data.datasets.utils import FileInfo, ImageTform, download_from_gdrive
from bolts.data.datasets.vision.base import PBVisionDataset

__all__ = ["CelebA", "CelebAttr", "CelebASplit"]


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


class CelebA(PBVisionDataset):
    """CelebA dataset."""

    transform: ImageTform
    """The data is downloaded to `download_dir` / `_BASE_FOLDER`."""
    _BASE_FOLDER: ClassVar[str] = "celeba"
    """The data is downloaded to `download_dir` / `_BASE_FOLDER` / `_IMAGE_DIR`."""
    _IMAGE_DIR: ClassVar[str] = "img_align_celeba"
    """Google drive IDs, MD5 hashes and filenames for the CelebA files."""
    _FILE_LIST: ClassVar[list[FileInfo]] = [
        FileInfo(
            name="img_align_celeba.zip",
            id="1zmsC4yvw-e089uHXj5EdP0BSZ0AlDQRR",
            md5="00d2c5bc6d35e252742224ab0c1e8fcb",
        ),
        FileInfo(
            name="list_attr_celeba.txt",
            id="1gxmFoeEPgF9sT65Wpo85AnHl3zsQ4NvS",
            md5="75e246fa4810816ffd6ee81facbd244c",
        ),
        FileInfo(
            name="list_eval_partition.txt",
            id="1ih_VMokoI774ErNWrb26lDeWlanUBpnX",
            md5="d32c9cbf5e040fd4025c592c306e6668",
        ),
    ]

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        download: bool = True,
        superclass: Union[CelebAttr, str] = CelebAttr.Smiling,
        subclass: Union[CelebAttr, str] = CelebAttr.Male,
        transform: Optional[ImageTform] = None,
        split: Optional[Union[CelebASplit, str]] = None,
    ) -> None:

        if isinstance(superclass, str):
            superclass = str_to_enum(str_=superclass, enum=CelebAttr)
        if isinstance(subclass, str):
            subclass = str_to_enum(str_=subclass, enum=CelebAttr)
        if isinstance(split, str):
            split = str_to_enum(str_=split, enum=CelebASplit)

        self.root = Path(root)
        self._image_dir = self.root / self._BASE_FOLDER
        self._base_dir = self.root / self._BASE_FOLDER
        image_dir = self._base_dir / self._IMAGE_DIR
        self.superclass = superclass
        self.subclass = subclass

        if download:
            download_from_gdrive(file_info=self._FILE_LIST, root=self._base_dir, logger=self.logger)
        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data don't exist at location {self._base_dir.resolve()}. Have you downloaded it?"
            )

        if split is None:
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
            skiprows = (splits != split.value).nonzero()[0] + 2
        attrs = pd.read_csv(
            self._base_dir / "list_attr_celeba.txt",
            delim_whitespace=True,
            header=1,
            usecols=[superclass.name, subclass.name],
            skiprows=skiprows,
        )

        x = np.array(attrs.index)
        s = torch.as_tensor(attrs[subclass.name].to_numpy())
        y = torch.as_tensor(attrs[superclass.name].to_numpy())
        # map from {-1, 1} to {0, 1}
        s = torch.div(s + 1, 2, rounding_mode='floor')
        y = torch.div(s + 1, 2, rounding_mode='floor')

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=image_dir)

    def _check_unzipped(self) -> bool:
        return (self._base_dir / self._IMAGE_DIR).is_dir()
