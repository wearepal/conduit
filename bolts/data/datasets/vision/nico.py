from __future__ import annotations
import logging
from pathlib import Path
from typing import ClassVar, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown
from kit import implements
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
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
            self.metadata = self.metadata[self.metadata["superclass"] == superclass]
        # # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # # substantially slower than indexing with numpy/torch
        self.x = self.metadata["filepath"].to_numpy()
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

        if not self._base_dir.with_suffix(".zip").exists():
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
        if ext not in [".zip", ".7z"] and check_integrity(str(fpath), self._MD5):
            raise RuntimeError('Dataset corrupted; try deleting it and redownloading it.')

    def _extract_metadata(self) -> None:
        """Extract concept/context/superclass information from the image filepaths and it save to csv."""
        image_paths: list[Path] = []
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

    def train_test_split(
        self,
        default_train_prop: float,
        train_props: dict[str | int, dict[str | int, float]] | None = None,
        seed: int | None = None,
    ) -> tuple[Subset, Subset]:
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
        test_inds = list(set(range(len(self))) - set(train_inds))

        return Subset(self, indices=train_inds), Subset(self, test_inds)

    @implements(VisionDataset)
    def __len__(self) -> int:
        return len(self.x)

    @implements(VisionDataset)
    def __getitem__(self, index: int) -> TernarySample:
        image = load_image(self._base_dir / self.x[index], backend=self._il_backend)
        image = apply_image_transform(image=image, transform=self.transform)
        target = self.y[index]
        return TernarySample(x=image, s=self.s[index], y=target)


def _preprocess_nico(path: Path) -> None:
    """
    Preprocess the original NICO data.
    This preprocessing entails two things:
        1) Renaming the files according to their concept/context/order of appearance.
        This renaming was necessary as some of the file-names were too long
        to be unzipped.
        2) Converting any GIFs into JPEGs by extracting the first frame.
    Note that this preprocessing will be performed **in-place**, overwriting the original data.
    """
    from PIL import Image

    for superclass in ("animals", "vehicles"):
        superclass_dir = path / superclass
        for class_dir in superclass_dir.glob("*"):
            for context_dir in class_dir.glob("*"):
                images_paths: list[Path] = []
                for ext in ("jpg", "jpeg", "png", "gif"):
                    images_paths.extend(context_dir.glob(f"**/*.{ext}"))
                for counter, image_path in enumerate(images_paths):
                    if image_path.suffix == ".gif":
                        image = Image.open(image_path).convert("RGBA")
                        new_image_path = image_path.with_suffix(".jpg")

                        background = Image.new('RGBA', image.size, (255, 255, 255))
                        new_image = Image.alpha_composite(background, image).convert("RGB")
                        new_image.save(new_image_path, "JPEG")
                        image_path.unlink()
                        image_path = new_image_path

                    concept = image_path.parent.parent.stem
                    context = image_path.parent.stem
                    new_name = (
                        image_path.parent / f"{concept}_{context}_{counter:04}.{image_path.suffix}"
                    )
                    image_path.rename(new_name)
