"""CelebA Dataset."""
from __future__ import annotations
from enum import Enum, auto
import logging
from pathlib import Path
from typing import ClassVar, Optional

import gdown
from kit import implements
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import VisionDataset

from bolts.data.datasets.utils import (
    ImageLoadingBackend,
    ImageTform,
    apply_image_transform,
    infer_il_backend,
    load_image,
)
from bolts.data.structures import TernarySample

__all__ = ["CelebA", "CelebAttr", "CelebASplit"]

LOGGER = logging.getLogger(__name__)


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


class CelebA(VisionDataset):
    """Celeba dataset."""

    transform: ImageTform
    """The data is downloaded to `download_dir` / `_BASE_FOLDER`."""
    _BASE_FOLDER: ClassVar[str] = "celeba"
    """Google drive IDs, MD5 hashes and filenames for the CelebA files."""
    _FILE_LIST: ClassVar[list[tuple[str, str, str]]] = [
        (
            "1zmsC4yvw-e089uHXj5EdP0BSZ0AlDQRR",  # File ID
            "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
            "img_align_celeba.zip",  # Filename
        ),
        (
            "1gxmFoeEPgF9sT65Wpo85AnHl3zsQ4NvS",
            "75e246fa4810816ffd6ee81facbd244c",
            "list_attr_celeba.txt",
        ),
        (
            "1ih_VMokoI774ErNWrb26lDeWlanUBpnX",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    def __init__(
        self,
        root: str,
        download: bool = True,
        superclass: CelebAttr = CelebAttr.Smiling,
        subclass: CelebAttr = CelebAttr.Male,
        transform: Optional[ImageTform] = None,
        split: Optional[CelebASplit] = None,
    ) -> None:
        super().__init__(root=root, transform=transform)

        self.base = Path(root) / self._BASE_FOLDER
        self._img_dir = self.base / "img_align_celeba"
        self.superclass = superclass
        self.subclass = subclass

        if download:
            self._download_and_unzip_data()
        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data don't exist at location {self.base.resolve()}. " "Have you downloaded it?"
            )

        if split is None:
            skiprows = None
        else:
            # splits: information about which samples belong to train, val or test
            splits = (
                pd.read_csv(
                    self.base / "list_eval_partition.txt", delim_whitespace=True, names=["split"]
                )
                .to_numpy()
                .squeeze()
            )
            skiprows = (splits != split.value).nonzero()[0] + 2
        attrs = pd.read_csv(
            self.base / "list_attr_celeba.txt",
            delim_whitespace=True,
            header=1,
            usecols=[superclass.name, subclass.name],
            skiprows=skiprows,
        )

        self.x = np.array(attrs.index)
        self.s = torch.tensor(attrs[subclass.name])
        self.y = torch.tensor(attrs[superclass.name])
        # map from {-1, 1} to {0, 1}
        self.s = torch.div(self.s + 1, 2, rounding_mode='floor')
        self.y = torch.div(self.s + 1, 2, rounding_mode='floor')

        # infer the appropriate image-loading backend based on the type of 'transform'
        self._il_backend: ImageLoadingBackend = infer_il_backend(self.transform)

    def _check_unzipped(self) -> bool:
        return self._img_dir.is_dir()

    def _download_and_unzip_data(self) -> None:
        """Attempt to download data if files cannot be found in the base directory."""
        if self._check_unzipped():
            LOGGER.info("Files already downloaded and unzipped.")
            return

        # Create the specified base directory if it doesn't already exist
        self.base.mkdir(parents=True, exist_ok=True)
        # -------------------------- Download the data ---------------------------
        LOGGER.info("Downloading the data from Google Drive.")
        for file_id, md5, filename in self._FILE_LIST:
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={file_id}",
                path=str(self.base / filename),
                quiet=False,
                md5=md5,
                postprocess=gdown.extractall if filename.endswith(".zip") else None,
            )

        if not self._check_unzipped():
            raise RuntimeError("Data seems to be downloaded but not unzipped. Download again?")

    @implements(VisionDataset)
    def __len__(self) -> int:
        return len(self.x)

    @implements(VisionDataset)
    def __getitem__(self, index: int) -> TernarySample:
        image = load_image(self._img_dir / self.x[index], backend=self._il_backend)
        image = apply_image_transform(image=image, transform=self.transform)
        target = self.y[index]
        return TernarySample(x=image, s=self.s[index], y=target)
