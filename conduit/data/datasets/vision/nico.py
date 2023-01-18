"""NICO Dataset."""
from enum import auto
from pathlib import Path
from typing import ClassVar, List, Optional, Union, cast

from PIL import Image, UnidentifiedImageError
import pandas as pd
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.datasets.utils import GdriveFileInfo, download_from_gdrive
from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = [
    "NICO",
    "NicoSuperclass",
]


class NicoSuperclass(StrEnum):
    ANIMALS = auto()
    VEHICLES = auto()


SampleType: TypeAlias = TernarySample


class NICO(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """Datset for Non-I.I.D. image classification introduced in
    'Towards Non-I.I.D. Image Classification: A Dataset and Baselines'
    """

    SampleType: TypeAlias = TernarySample
    Superclass: TypeAlias = NicoSuperclass

    _FILE_INFO: ClassVar[GdriveFileInfo] = GdriveFileInfo(
        name="NICO.zip",
        id="1L6cHNhuwwvrolukBklFyhFu7Y8WUUIQ7",
        md5="78c686f84e31ad6b6c052f97ed5f532b",
    )

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        superclass: Optional[Union[NicoSuperclass, str]] = NicoSuperclass.ANIMALS,
    ) -> None:
        self.superclass = NicoSuperclass(superclass) if isinstance(superclass, str) else superclass
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
        self.class_tree = (
            self.metadata[["concept", "context"]]
            .drop_duplicates()
            .groupby("concept")
            .agg(set)
            .to_dict()["context"]
        )
        self.concept_label_decoder = (
            self.metadata[["concept", "concept_le"]].set_index("concept_le").to_dict()["concept"]
        )
        self.context_label_decoder = (
            self.metadata[["context", "context_le"]].set_index("context_le").to_dict()["context"]
        )

        if self.superclass is not None:
            self.metadata = self.metadata[self.metadata["superclass"] == str(self.superclass)]
        # # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # # substantially slower than indexing with numpy/torch
        x = self.metadata["filepath"].to_numpy()
        y = torch.as_tensor(self.metadata["concept_le"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["context_le"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)

    def _check_unzipped(self) -> bool:
        return all((self._base_dir / sc.name).exists() for sc in NicoSuperclass)

    def _extract_metadata(self) -> None:
        """Extract concept/context/superclass information from the image filepaths and it save to csv."""
        self.logger.info("Extracting metadata.")
        image_paths: List[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            image_paths.extend(self._base_dir.glob(f"**/*.{ext}"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str)
        metadata = cast(
            pd.DataFrame,
            filepaths.str.split("/", expand=True).rename(
                columns={0: "superclass", 1: "concept", 2: "context", 3: "filename"}
            ),
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


def _gif_to_jpeg(image: Image.Image) -> Image.Image:
    background = Image.new('RGBA', image.size, (255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def preprocess_nico(path: Path) -> None:
    """
    Preprocess the original NICO data.
    This preprocessing entails two things:
        1) Renaming the files according to their concept/context/order of appearance.
        This renaming was necessary as some of the file-names were too long
        to be unzipped.
        2) Converting any GIFs into JPEGs by extracting the first frame.
    Note that this preprocessing will be performed **in-place**, overwriting the original data.
    """
    for superclass in ("animals", "vehicles"):
        superclass_dir = path / superclass
        for class_dir in superclass_dir.glob("*"):
            for context_dir in class_dir.glob("*"):
                images_paths: List[Path] = []
                for ext in ("jpg", "jpeg", "png", "gif"):
                    images_paths.extend(context_dir.glob(f"**/*.{ext}"))
                for counter, image_path in enumerate(images_paths):
                    try:
                        image = Image.open(image_path)
                        if image.format == "GIF":
                            image = image.convert("RGBA")
                            # Convert from gif to jpeg by extracting the first frame
                            new_image = _gif_to_jpeg(image)
                            new_image_path = image_path.with_suffix(".jpg")
                            # Delete the original gif
                            image_path.unlink()
                            new_image.save(new_image_path, "JPEG")
                            assert new_image_path.exists()
                            image_path = new_image_path

                        concept = image_path.parent.parent.stem
                        context = image_path.parent.stem
                        new_name = (
                            image_path.parent
                            / f"{concept}_{context}_{counter:04}{image_path.suffix}".replace(
                                " ", "_"
                            )
                        )
                        image_path.rename(new_name)
                    # Image is corrupted - delete it
                    except UnidentifiedImageError:
                        image_path.unlink()
