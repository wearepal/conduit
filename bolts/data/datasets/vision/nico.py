from __future__ import annotations
import logging
from pathlib import Path
from typing import ClassVar, List, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown
import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from typing_extensions import Literal, get_args

from bolts.data.datasets.utils import (
    ImageLoadingBackend,
    ImageTform,
    TernarySample,
    apply_image_transform,
    infer_il_backend,
    load_image,
)

__all__ = ["NICO"]

LOGGER = logging.getLogger(__name__)

NicoSuperclass = Literal["animals", "vehicles"]


class NICO(VisionDataset):
    """Datset for Non-I.I.D. image classification introduced in
    'Towards Non-I.I.D. Image Classification: A Dataset and Baselines'
    """

    _FILE_ID: ClassVar[str] = "1RlspK4FkbrvZEzh-tyXBJMZyvs1DM0cP"  # File ID
    _MD5: ClassVar[str] = "6f21e6484fec0b3a8ef87f0d3115ce93"  # MD5 checksum
    _BASE_FOLDER: ClassVar[str] = "NICO"

    transform: ImageTform

    def __init__(
        self,
        root: str,
        download: bool = True,
        transform: ImageTform = A.Compose([A.Normalize(), ToTensorV2()]),
        superclass: NicoSuperclass | None = "animals",
    ) -> None:
        super().__init__(root=root, transform=transform)

        self.root: Path = Path(self.root)
        self.download = download
        self._base_dir = self.root / self._BASE_FOLDER
        self._metadata_path = self._base_dir / "metadata.csv"
        self.superclass = superclass

        if self.download:
            self._download_and_unzip_data()
        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data don't exist at location {self._base_dir.resolve()}. "
                "Have you downloaded it?"
            )

        self.metadata = pd.read_csv(self._base_dir / "metadata.csv")
        if superclass is not None:
            self.metadata = self.metadata[self.metadata["superclass"] == superclass]
        # # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # # substantially slower than indexing with numpy/torch
        self.x = self.metadata["filepath"].values
        self.s = torch.as_tensor(self.metadata["context_le"], dtype=torch.int32)
        self.y = torch.as_tensor(self.metadata["concept_le"], dtype=torch.int32)

        self._il_backend: ImageLoadingBackend = infer_il_backend(self.transform)

    def _check_unzipped(self) -> bool:
        return all((self._base_dir / sc).exists() for sc in get_args(NicoSuperclass))

    def _download_and_unzip_data(self) -> None:
        """Attempt to download data if files cannot be found in the root directory."""

        if self._check_unzipped():
            LOGGER.info("Files already downloaded and unzipped.")
            return

        if self._base_dir.with_suffix(".zip").exists():
            self._check_integrity()
        else:

            # Create the specified root directory if it doesn't already exist
            self.root.mkdir(parents=True, exist_ok=True)
            # -------------------------- Download the data ---------------------------
            LOGGER.info("Downloading the data from Google Drive.")
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={self._FILE_ID}",
                path=self._base_dir.with_suffix(".zip"),
                quiet=False,
                md5=self._MD5,
            )
            self._check_integrity()
        # ------------------------------ Unzip the data ------------------------------
        import zipfile

        LOGGER.info("Unzipping the data; this may take a while.")
        with zipfile.ZipFile(self._base_dir.with_suffix(".zip"), "r") as fhandle:
            fhandle.extractall(str(self.root))

        if not self._metadata_path.exists():
            self._extract_metadata()

    def _check_integrity(self) -> None:
        from torchvision.datasets.utils import check_integrity

        fpath = self._base_dir.with_suffix(".zip")
        ext = fpath.suffix
        if not ext in [".zip", ".7z"] and check_integrity(str(fpath), self._MD5):
            raise RuntimeError('Dataset corrupted; try deleting it and redownloading it.')

    def _extract_metadata(self) -> None:
        """Extract concept/context/superclass information from the image filepaths and it save to csv."""
        images: List[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            images.extend(self._base_dir.glob(f"**/*.{ext}"))
        images = [str(image.relative_to(self._base_dir)) for image in images]
        filepaths = pd.Series(images)
        metadata = cast(
            pd.DataFrame,
            filepaths.str.split("/", expand=True).rename(
                columns={0: "superclass", 1: "concept", 2: "context", 3: "filename"}
            ),
        )
        metadata["filepath"] = filepaths
        metadata.sort_index(axis=1, inplace=True)
        metadata.sort_index(inplace=True)
        metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path)

    def _label_encode_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Label encode the extracted concept/context/superclass information."""
        for col in metadata.columns:
            # Skip over filepath and filename columns - these do not metadata
            if "file" in col:
                continue
            # Add a new column containing the label-encoded data
            metadata[f"{col}_le"] = metadata[col].factorize()[0]
        return metadata

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> TernarySample:
        image = load_image(self._base_dir / self.x[index], backend=self._il_backend)
        image = apply_image_transform(image=image, transform=self.transform)
        target = self.y[index]
        return TernarySample(x=image, s=self.s[index], y=target)
