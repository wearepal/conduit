"""ISIC Dataset."""
from __future__ import annotations
from enum import auto
from itertools import islice
import os
from pathlib import Path
import shutil
from typing import ClassVar, Iterable, Iterator, List, Optional, TypeVar, Union
import zipfile

from PIL import Image
import pandas as pd
from ranzen import StrEnum, flatten_dict
from ranzen.decorators import parsable
import requests
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import TypeAlias

from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["IsicAttr", "ISIC"]


class IsicAttr(StrEnum):
    HISTO = auto()
    MALIGNANT = auto()
    PATCH = auto()


T = TypeVar("T")
SampleType: TypeAlias = TernarySample


class ISIC(CdtVisionDataset[SampleType, Tensor, Tensor]):
    """PyTorch Dataset for the ISIC 2018 dataset from
    'Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International
    Skin Imaging Collaboration (ISIC)',"""

    SampleType: TypeAlias = TernarySample
    Attr: TypeAlias = IsicAttr

    LABELS_FILENAME: ClassVar[str] = "labels.csv"
    METADATA_FILENAME: ClassVar[str] = "metadata.csv"
    _PBAR_COL: ClassVar[str] = "#fac000"
    _REST_API_URL: ClassVar[str] = "https://isic-archive.com/api/v1"

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        max_samples: int = 25_000,  # default is the number of samples used for the NSLB paper
        target_attr: IsicAttr = IsicAttr.MALIGNANT,
        context_attr: IsicAttr = IsicAttr.HISTO,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.target_attr = IsicAttr(target_attr) if isinstance(target_attr, str) else target_attr
        self.context_attr = (
            IsicAttr(context_attr) if isinstance(context_attr, str) else context_attr
        )

        self.root = Path(root)
        self.download = download
        self._base_dir = self.root / self.__class__.__name__
        self._processed_dir = self._base_dir / "processed"
        self._raw_dir = self._base_dir / "raw"

        if max_samples < 1:
            raise ValueError("max_samples must be a positive integer.")
        self.max_samples = max_samples
        if self.download:
            self._download_data()
            self._preprocess_data()
        elif not self._check_downloaded():
            raise FileNotFoundError(
                f"Data not found at location {self._processed_dir.resolve()}. "
                "Have you downloaded it?"
            )

        self.metadata = pd.read_csv(self._processed_dir / self.LABELS_FILENAME)
        # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # considerably slower than indexing with numpy/torch
        x = self.metadata["path"].to_numpy()
        s = torch.as_tensor(self.metadata[str(self.context_attr)], dtype=torch.int32)
        y = torch.as_tensor(self.metadata[str(self.target_attr)], dtype=torch.int32)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._processed_dir)

    def _check_downloaded(self) -> bool:
        return (self._raw_dir / "images").exists() and (
            self._raw_dir / self.METADATA_FILENAME
        ).exists()

    def _check_processed(self) -> bool:
        return (self._processed_dir / "ISIC-images").exists() and (
            self._processed_dir / self.LABELS_FILENAME
        ).exists()

    @staticmethod
    def chunk(it: Iterable[T], *, size: int) -> Iterator[List[T]]:
        """Divide any iterable into chunks of the given size."""
        it = iter(it)
        return iter(lambda: list(islice(it, size)), [])  # this is magic from stackoverflow

    def _download_isic_metadata(self) -> pd.DataFrame:
        """Downloads the metadata CSV from the ISIC website."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        req = requests.get(
            f"{self._REST_API_URL}/image?limit={self.max_samples}&sort=name&sortdir=1&detail=false"
        )
        image_ids = req.json()
        image_ids = [image_id["_id"] for image_id in image_ids]

        template_start = "?limit=300&sort=name&sortdir=1&detail=true&imageIds=%5B%22"
        template_sep = "%22%2C%22"
        template_end = "%22%5D"
        entries = []
        with tqdm(
            total=(len(image_ids) - 1) // 300 + 1,
            desc="Downloading metadata",
            colour=self._PBAR_COL,
        ) as pbar:
            for block in self.chunk(image_ids, size=300):
                pbar.set_postfix(image_id=block[0])
                args = ""
                args += template_start
                args += template_sep.join(block)
                args += template_end
                req = requests.get(f"{self._REST_API_URL}/image{args}")
                image_details = req.json()
                for image_detail in image_details:
                    entry = flatten_dict(image_detail, sep=".")
                    entries.append(entry)
                pbar.update()

        metadata_df = pd.DataFrame(entries)
        metadata_df = metadata_df.set_index("_id")
        metadata_df.to_csv(self._raw_dir / self.METADATA_FILENAME)
        return metadata_df

    def _download_isic_images(self) -> None:
        """Given the metadata CSV, downloads the ISIC images."""
        metadata_path = self._raw_dir / self.METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{self.METADATA_FILENAME} not downloaded. "
                "Run 'download_isic_data` before this function."
            )
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df.set_index("_id")

        template_start = "?include=images&imageIds=%5B%22"
        template_sep = "%22%2C%22"
        template_end = "%22%5D"
        raw_image_dir = self._raw_dir / "images"
        raw_image_dir.mkdir(exist_ok=True)
        image_ids = list(metadata_df.index)
        with tqdm(
            total=(len(image_ids) - 1) // 50 + 1, desc="Downloading images", colour=self._PBAR_COL
        ) as pbar:
            for i, block in enumerate(self.chunk(image_ids, size=50)):
                pbar.set_postfix(image_id=block[0])
                args = ""
                args += template_start
                args += template_sep.join(block)
                args += template_end
                req = requests.get(f"{self._REST_API_URL}/image/download{args}", stream=True)
                req.raise_for_status()
                image_path = raw_image_dir / f"{i}.zip"
                with open(image_path, "wb") as f:
                    shutil.copyfileobj(req.raw, f)
                del req
                pbar.update()

    def _preprocess_isic_metadata(self) -> None:
        """Preprocesses the raw ISIC metadata."""
        self._processed_dir.mkdir(exist_ok=True)

        metadata_path = self._raw_dir / self.METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{self.METADATA_FILENAME} not found while preprocessing ISIC dataset. "
                "Run `download_isic_metadata` and `download_isic_images` before "
                "calling `preprocess_isic_metadata`."
            )
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df.set_index("_id")
        labels_df = self._remove_uncertain_diagnoses(metadata_df)
        labels_df = self._add_patch_column(labels_df)

        labels_df["path"] = (
            str(self._processed_dir)
            + os.sep
            + "ISIC-images"
            + os.sep
            + labels_df["dataset.name"]
            + os.sep
            + labels_df["name"]
            + ".jpg"
        )
        labels_df.to_csv(self._processed_dir / self.LABELS_FILENAME)

    def _preprocess_isic_images(self) -> None:
        """Preprocesses the images."""
        if (self._processed_dir / "ISIC-images").is_dir():
            return
        if not (self._raw_dir / "images").is_dir():
            raise FileNotFoundError(
                "Raw ISIC images not found. Run `download_isic_images` before "
                "calling `preprocess_isic_images`."
            )
        labels_df = pd.read_csv(self._processed_dir / self.LABELS_FILENAME)
        labels_df = labels_df.set_index("_id")

        self._processed_dir.mkdir(exist_ok=True)
        image_zips = tuple((self._raw_dir / "images").glob("**/*.zip"))
        with tqdm(total=len(image_zips), desc="Unzipping images", colour=self._PBAR_COL) as pbar:
            for file in image_zips:
                pbar.set_postfix(file_index=file.stem)
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(self._processed_dir)
                    pbar.update()
        images: List[Path] = []
        for ext in ("jpg", "jpeg", "png", "gif"):
            images.extend(self._processed_dir.glob(f"**/*.{ext}"))
        with tqdm(total=len(images), desc="Processing images", colour=self._PBAR_COL) as pbar:
            for image_path in images:
                pbar.set_postfix(image_name=image_path.stem)
                image = Image.open(image_path)
                image = image.resize((224, 224))  # Resize the images to be of size 224 x 224
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(str(image_path.rename(image_path.with_suffix(".jpg"))))
                pbar.update()

    @staticmethod
    def _remove_uncertain_diagnoses(metadata_df: pd.DataFrame) -> pd.DataFrame:
        labels_df = metadata_df.loc[
            metadata_df["meta.clinical.benign_malignant"].isin({"benign", "malignant"})
        ]  # throw out unknowns
        malignant_mask = labels_df["meta.clinical.benign_malignant"] == "malignant"
        labels_df["malignant"] = malignant_mask.astype("uint8")

        labels_df["meta.clinical.diagnosis_confirm_type"].fillna(
            value="non-histopathology", inplace=True
        )
        histopathology_mask = labels_df["meta.clinical.diagnosis_confirm_type"] == "histopathology"
        labels_df["histo"] = histopathology_mask.astype("uint8")

        return labels_df

    @staticmethod
    def _add_patch_column(labels_df: pd.DataFrame) -> pd.DataFrame:
        """Adds a patch column to the input DataFrame."""
        patch_mask = labels_df["dataset.name"] == "SONIC"
        # add to labels_df
        labels_df["patch"] = None
        labels_df.loc[patch_mask, "patch"] = 1
        labels_df.loc[~patch_mask, "patch"] = 0
        assert all(patch is not None for patch in labels_df["patch"])
        return labels_df

    def _download_data(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""
        # # Check whether the data has already been downloaded - if it has and the integrity
        # # of the files can be confirmed, then we are done
        if self._check_downloaded():
            self.logger.info("Files already downloaded and verified.")
            return
        # Create the directory and any required ancestors if not already existent
        self._base_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Downloading metadata into {self._raw_dir / self.METADATA_FILENAME}...")
        self._download_isic_metadata()
        self.logger.info(
            f"Downloading data into {self._raw_dir} for up to {self.max_samples} samples..."
        )
        self._download_isic_images()

    def _preprocess_data(self) -> None:
        """Preprocess the downloaded data if the processed image-directory/metadata don't exist."""
        # If the data has already been processed, skip this operation
        if self._check_processed():
            self.logger.info("Metadata and images already preprocessed.")
            return
        self.logger.info(
            "Preprocessing metadata (adding columns, removing uncertain diagnoses) and saving into"
            f" {str(self._processed_dir / self.LABELS_FILENAME)}..."
        )
        self._preprocess_isic_metadata()
        self.logger.info(
            "Preprocessing images (transforming to 3-channel RGB, resizing to 224x224) and saving "
            f"into{str(self._processed_dir / 'ISIC-images')}..."
        )
        self._preprocess_isic_images()
