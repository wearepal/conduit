from __future__ import annotations
import logging
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ethicml import CELEBA_BASE_FOLDER, CELEBA_FILE_LIST, CelebAttrs
import ethicml as em
import gdown
import torch
from torchvision.datasets import VisionDataset

from bolts.data.datasets.utils import (
    ImageLoadingBackend,
    ImageTform,
    TernarySample,
    apply_image_transform,
    infer_il_backend,
    load_image,
)

__all__ = ["Celeba", "CelebAttrs"]

LOGGER = logging.getLogger(__name__)


class Celeba(VisionDataset):
    """Celeba dataset."""

    transform: ImageTform

    def __init__(
        self,
        root: str,
        download: bool = True,
        transform: ImageTform = A.Compose([A.Normalize(), ToTensorV2()]),
        superclass: CelebAttrs = "Smiling",
        subclass: CelebAttrs = "Male",
    ) -> None:
        self.base = Path(root) / CELEBA_BASE_FOLDER
        super().__init__(root=str(self.base), transform=transform)

        self.download = download
        self.superclass = superclass
        self.subclass = subclass
        self._img_dir = self.base / "img_align_celeba"

        if self.download:
            self._download_and_unzip_data()
        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data don't exist at location {self.base.resolve()}. " "Have you downloaded it?"
            )

        # load meta data
        dataset, _ = em.celeba(
            download_dir=root,
            label=superclass,
            sens_attr=subclass,
            download=False,
            check_integrity=False,
        )
        assert dataset is not None, "could not load CelebA"
        data_tup = dataset.load(labels_as_features=True)
        self.metadata = data_tup.x

        self.x = self.metadata["filename"].to_numpy(copy=True)
        self.s = torch.as_tensor(data_tup.s, dtype=torch.int32)
        self.y = torch.as_tensor(data_tup.y, dtype=torch.int32)

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
        for file_id, md5, filename in CELEBA_FILE_LIST:
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={file_id}",
                path=str(self.base / filename),
                quiet=False,
                md5=md5,
            )
        # ------------------------------ Unzip the data ------------------------------
        import zipfile

        LOGGER.info("Unzipping the data; this may take a while.")
        with zipfile.ZipFile(self.base / "img_align_celeba.zip", "r") as fhandle:
            fhandle.extractall(str(self.base))

        assert self._check_unzipped()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> TernarySample:
        image = load_image(self._img_dir / self.x[index], backend=self._il_backend)
        image = apply_image_transform(image=image, transform=self.transform)
        target = self.y[index]
        return TernarySample(x=image, s=self.s[index], y=target)
