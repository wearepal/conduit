"""NICO Dataset."""
from __future__ import annotations
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Optional, Union, cast

from PIL import Image, UnidentifiedImageError
from kit import parsable, str_to_enum
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from conduit.data.datasets.utils import FileInfo, ImageTform, download_from_gdrive
from conduit.data.structures import TrainTestSplit

from .base import CdtVisionDataset

__all__ = ["NICO", "NicoSuperclass"]


class NicoSuperclass(Enum):
    animals = auto()
    vehicles = auto()


class NICO(CdtVisionDataset):
    """Datset for Non-I.I.D. image classification introduced in
    'Towards Non-I.I.D. Image Classification: A Dataset and Baselines'
    """

    _FILE_INFO: ClassVar[FileInfo] = FileInfo(
        name="NICO.zip",
        id="1L6cHNhuwwvrolukBklFyhFu7Y8WUUIQ7",
        md5="78c686f84e31ad6b6c052f97ed5f532b",
    )
    _BASE_FOLDER: ClassVar[str] = "NICO"

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = True,
        transform: Optional[ImageTform] = None,
        superclass: Optional[Union[NicoSuperclass, str]] = NicoSuperclass.animals,
    ) -> None:

        if isinstance(superclass, str):
            superclass = str_to_enum(str_=superclass, enum=NicoSuperclass)
        self.root = Path(root)
        self.download = download
        self._base_dir = self.root / self._BASE_FOLDER
        self._metadata_path = self._base_dir / "metadata.csv"
        assert isinstance(superclass, NicoSuperclass) or superclass is None
        self.superclass = superclass

        if self.download:
            download_from_gdrive(file_info=self._FILE_INFO, root=self.root, logger=self.logger)
        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data not found at location {self._base_dir.resolve()}. " "Have you downloaded it?"
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

        if superclass is not None:
            self.metadata = self.metadata[self.metadata["superclass"] == superclass.name]
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
        self.log("Extracting metadata.")
        image_paths: list[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            image_paths.extend(self._base_dir.glob(f"**/*.{ext}"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str)
        metadata = cast(
            pd.DataFrame,
            filepaths.str.split("/", expand=True).rename(  # type: ignore[attr-defined]
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

    def train_test_split(
        self,
        default_train_prop: float,
        *,
        train_props: dict[str | int, dict[str | int, float]] | None = None,
        seed: int | None = None,
    ) -> TrainTestSplit:
        """Split the data into train/test sets with the option to condition on concept/context."""
        # Initialise the random-number generator
        rng = np.random.default_rng(seed)
        # List to store the indices of the samples apportioned to the train set
        # - those for the test set will be computed by complement
        train_inds: list[int] = []
        # Track which indices have been sampled for either split
        unvisited = np.ones(len(self), dtype=np.bool_)

        def _sample_train_inds(
            _mask: np.ndarray,
            *,
            _context: str | int | None = None,
            _concept: str | None = None,
            _train_prop: float = default_train_prop,
        ) -> list[int]:
            if _context is not None and _concept is None:
                raise ValueError("Concept must be specified if context is.")
            if _context is not None:
                # Allow the context to be speicifed either by its name or its label-encoding
                _context = (
                    self.context_label_decoder(_context) if isinstance(_context, int) else _context
                )
                if _context not in self.class_tree[_concept]:
                    raise ValueError(
                        f"'{_context}' is not a valid context for concept '{_concept}'."
                    )
                # Condition the mask on the context
                _mask = _mask & (self.metadata["context"] == _context).to_numpy()
            # Compute the overall size of the concept/context subset
            _subset_size = np.count_nonzero(_mask)
            # Compute the size of the train split
            _train_subset_size = round(_train_prop * _subset_size)
            # Sample the train indices (without replacement)
            _train_inds = rng.choice(
                np.nonzero(_mask)[0], size=_train_subset_size, replace=False
            ).tolist()
            # Mark the sampled indices as 'visited'
            unvisited[_mask] = False

            return _train_inds

        if train_props is not None:
            for concept, value in train_props.items():
                # Allow the concept to be speicifed either by its name or its label-encoding
                concept = (
                    self.concept_label_decoder[concept] if isinstance(concept, int) else concept
                )
                if concept not in self.class_tree.keys():
                    raise ValueError(f"'{concept}' is not a valid concept.")
                concept_mask = (self.metadata["concept"] == concept).to_numpy()
                # Specifying proportions at the context/concept level, rather than concept-wide
                if isinstance(value, dict):
                    for context, train_prop in value.items():
                        train_inds.extend(
                            _sample_train_inds(
                                _mask=concept_mask,
                                _concept=concept,
                                _context=context,
                                _train_prop=train_prop,
                            )
                        )
                # Split at the class level (without conditioning on contexts)
                else:
                    train_inds.extend(
                        _sample_train_inds(_mask=concept_mask, _context=None, _train_prop=value)
                    )
        # Apportion any remaining samples to the training set using default_train_prop
        train_inds.extend(_sample_train_inds(_mask=unvisited, _train_prop=default_train_prop))
        # Compute the test indices by complement of the train indices
        train_data = Subset(self, indices=train_inds)
        test_inds = list(set(range(len(self))) - set(train_inds))
        test_data = Subset(self, indices=test_inds)

        return TrainTestSplit(train=train_data, test=test_data)


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
                images_paths: list[Path] = []
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
