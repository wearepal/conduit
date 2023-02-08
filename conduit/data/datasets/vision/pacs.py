"""PACS Dataset."""
from enum import auto
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import numpy as np
import pandas as pd
from ranzen import StrEnum, parsable, str_to_enum
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias

from conduit.data.datasets.utils import GdriveFileInfo, download_from_gdrive
from conduit.data.structures import TernarySample, TrainTestSplit

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = [
    "PACS",
    "PacsDomain",
]


SampleType: TypeAlias = TernarySample


class PacsDomain(StrEnum):
    PHOTO = auto()
    ART_PAINTING = auto()
    CARTOON = auto()
    SKETCH = auto()


class PACS(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """
    PACS (photo (P), art painting (A), cartoon (C), and sketch (S)) dataset for evaluating domain
    generalisation as introduced in `PACS`_ 'Deeper, Broader and Artier Domain Generalization'.

    .. _PACS:
        https://arxiv.org/abs/1710.03077
    """

    SampleType: TypeAlias = TernarySample
    Domain: TypeAlias = PacsDomain

    _FILE_INFO: ClassVar[GdriveFileInfo] = GdriveFileInfo(
        name="PACS.zip",
        id="1lNOnDplyxhdDdyfb4KRIh0p0uom8aS1i",
        md5="5e43a6e01e53567923621ae5ce025f4e",
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        domains: Optional[Union[Domain, str, List[Union[str, Domain]]]] = None,
    ) -> None:
        if isinstance(domains, str):
            self.domains = PacsDomain(domains)
        elif isinstance(domains, list):
            domains_ = [PacsDomain(elem) for elem in domains]
            self.domains = domains_
        else:
            self.domains = domains

        self.root = Path(root)
        self.download = download
        self._base_dir = self.root / self.__class__.__name__
        self._metadata_path = self._base_dir / "metadata.csv"

        if self.download:
            download_from_gdrive(file_info=self._FILE_INFO, root=self.root, logger=self.logger)
        elif not self._check_unzipped():
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )
        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self._base_dir / "metadata.csv")

        # Label decoders (for mapping from the label encodings back to their original string
        # values).
        self.domain_label_decoder = (
            self.metadata[["domain", "domain_le"]].set_index("domain_le").to_dict()["domain"]
        )
        self.class_label_decoder = (
            self.metadata[["class", "class_le"]].set_index("class_le").to_dict()["class"]
        )

        if isinstance(self.domains, PacsDomain):
            self.metadata = self.metadata[self.metadata["domain"] == str(self.domains)]
        elif isinstance(self.domains, list):
            str_list = [str(elem) for elem in self.domains]
            self.metadata = self.metadata[self.metadata["domain"].isin(str_list)]

        # # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # # substantially slower than indexing with numpy/torch
        x = self.metadata["filepath"].to_numpy()
        y = torch.as_tensor(self.metadata["class_le"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["domain_le"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)

    def _check_unzipped(self) -> bool:
        return all((self._base_dir / str(sc)).exists() for sc in PacsDomain)

    def _extract_metadata(self) -> None:
        """Extract domain/class information from the image filepaths and it save to csv."""
        self.logger.info("Extracting metadata.")
        image_paths: List[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            image_paths.extend(self._base_dir.glob(f"**/*.{ext}"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str)
        metadata = filepaths.str.split("/", expand=True).rename(
            columns={0: "domain", 1: "class", 2: "filename"}
        )
        metadata["filepath"] = filepaths
        metadata.sort_index(axis=1, inplace=True)
        metadata.sort_values(by=["filepath"], axis=0, inplace=True)
        metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path)

    @staticmethod
    def _label_encode_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
        """Label encode the extracted concept/context/superclass information."""
        for col in metadata.columns:
            # Skip over filepath and filename columns
            if "file" not in col:
                # Add a new column containing the label-encoded data
                metadata[f"{col}_le"] = metadata[col].factorize()[0]
        return metadata

    def domain_split(
        self,
        target_domains: Union[PacsDomain, str, List[Union[str, PacsDomain]]],
    ) -> TrainTestSplit[Self]:
        if isinstance(target_domains, list):
            target_domains = [
                str(str_to_enum(str_=elem, enum=PacsDomain)) for elem in target_domains
            ]
            test_mask = self.metadata["domain"].isin(target_domains)
        else:
            target_domains = str_to_enum(str_=target_domains, enum=PacsDomain)
            test_mask = self.metadata["domain"] == str(target_domains)
        test_mask_np = test_mask.to_numpy()
        test_inds = test_mask_np.nonzero()[0].astype(dtype=np.int64)
        if len(test_inds) == 0:
            raise ValueError(f"No samples in dataset belonging to domain(s) '{target_domains}'.")
        train_inds = (~test_mask_np).nonzero()[0].astype(dtype=np.int64)
        if len(train_inds) == 0:
            raise ValueError(f"No samples in dataset from which to construct the source domain(s).")

        train_subset = self.subset(list(train_inds))
        test_subset = self.subset(list(test_inds))

        return TrainTestSplit(train=train_subset, test=test_subset)
