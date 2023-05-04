"""NICO Dataset."""
from enum import auto
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from ranzen import StrEnum, parsable
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["NICOPP", "NicoPPTarget", "NicoPPAttr"]


class NicoPPTarget(StrEnum):
    """Class labels in NICO++."""

    AIRPLANE = auto()
    BEAR = auto()
    BICYCLE = auto()
    BIRD = auto()
    BUS = auto()
    BUTTERFLY = auto()
    CACTUS = auto()
    CAR = auto()
    CAT = auto()
    CHAIR = auto()
    CLOCK = auto()
    CORN = auto()
    COW = auto()
    CRAB = auto()
    CROCODILE = auto()
    DOG = auto()
    DOLPHIN = auto()
    ELEPHANT = auto()
    FISHING_ROD = "fishing rod"  # class name contains space
    FLOWER = auto()
    FOOTBALL = auto()
    FOX = auto()
    FROG = auto()
    GIRAFFE = auto()
    GOOSE = auto()
    GUN = auto()
    HAT = auto()
    HELICOPTER = auto()
    HORSE = auto()
    HOT_AIR_BALLOON = "hot air balloon"  # class name contains space
    KANGAROO = auto()
    LIFEBOAT = auto()
    LION = auto()
    LIZARD = auto()
    MAILBOX = auto()
    MONKEY = auto()
    MOTORCYCLE = auto()
    OSTRICH = auto()
    OWL = auto()
    PINEAPPLE = auto()
    PUMPKIN = auto()
    RABBIT = auto()
    RACKET = auto()
    SAILBOAT = auto()
    SCOOTER = auto()
    SEAL = auto()
    SHEEP = auto()
    SHIP = auto()
    SHRIMP = auto()
    SPIDER = auto()
    SQUIRREL = auto()
    SUNFLOWER = auto()
    TENT = auto()
    TIGER = auto()
    TORTOISE = auto()
    TRAIN = auto()
    TRUCK = auto()
    UMBRELLA = auto()
    WHEAT = auto()
    WOLF = auto()


class NicoPPAttr(StrEnum):
    """Common attributes in NICO++."""

    AUTUMN = auto()
    DIM = auto()
    GRASS = auto()
    OUTDOOR = auto()
    ROCK = auto()
    WATER = auto()


SampleType: TypeAlias = TernarySample


class NICOPP(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """NICO++ dataset."""

    SampleType: TypeAlias = TernarySample
    Superclass: TypeAlias = NicoPPTarget
    Subclass: TypeAlias = NicoPPAttr

    less_than_75_samples: ClassVar[Dict[NicoPPTarget, Tuple[NicoPPAttr, ...]]] = {
        NicoPPTarget.CACTUS: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.CORN: (NicoPPAttr.ROCK,),
        NicoPPTarget.CRAB: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.CROCODILE: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.DOLPHIN: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.FOOTBALL: (NicoPPAttr.ROCK,),
        NicoPPTarget.HAT: (NicoPPAttr.ROCK,),
        NicoPPTarget.LIFEBOAT: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.LIZARD: (NicoPPAttr.DIM,),
        NicoPPTarget.OSTRICH: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.PINEAPPLE: (NicoPPAttr.AUTUMN, NicoPPAttr.DIM, NicoPPAttr.ROCK),
        NicoPPTarget.RACKET: (NicoPPAttr.AUTUMN, NicoPPAttr.ROCK),
        NicoPPTarget.SEAL: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.SHRIMP: (
            NicoPPAttr.AUTUMN,
            NicoPPAttr.DIM,
            NicoPPAttr.OUTDOOR,
            NicoPPAttr.ROCK,
        ),
        NicoPPTarget.SPIDER: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.SUNFLOWER: (NicoPPAttr.AUTUMN,),
        NicoPPTarget.WHEAT: (NicoPPAttr.ROCK,),
    }

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        transform: Optional[ImageTform] = None,
        superclasses: Optional[Sequence[Union[NicoPPTarget, str]]] = None,
    ) -> None:
        self.superclasses: Optional[List[NicoPPTarget]] = None
        if superclasses is not None:
            assert superclasses, "superclasses should be a non-empty list"
            self.superclasses = [NicoPPTarget(superclass) for superclass in superclasses]
        self.root = Path(root)
        self._base_dir = self.root / "nico_plus_plus" / "track_1" / "public_dg_0416" / "train"
        self._metadata_path = self._base_dir / "metadata.csv"

        if not self._check_unzipped():
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )
        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self._base_dir / "metadata.csv")

        if self.superclasses is not None:
            self.metadata = self.metadata[self.metadata["superclass"].isin(self.superclasses)]
        # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # substantially slower than indexing with numpy/torch
        x = self.metadata["filepath"].to_numpy()
        y = torch.as_tensor(self.metadata["superclass_le"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["subclass_le"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)

    @property
    def default_train_props(self) -> Dict[int, Dict[int, float]]:
        """Zero out the (s,y) pairs which have fewer than 75 samples."""
        atoi = self.subclass_label_encoder
        ytoi = self.superclass_label_encoder
        return {
            ytoi[target]: {atoi[attr]: 0.0 for attr in attrs}
            for target, attrs in self.less_than_75_samples.items()
        }

    @cached_property
    def class_tree(self) -> Dict[str, Set[str]]:
        return (
            self.metadata[["superclass", "subclass"]]
            .drop_duplicates()
            .groupby("superclass")
            .agg(set)
            .to_dict()["subclass"]
        )

    @cached_property
    def superclass_label_encoder(self) -> Dict[NicoPPTarget, int]:
        return dict(
            (NicoPPTarget(name), val) for name, val in self._get_label_mapping("superclass")
        )

    @cached_property
    def subclass_label_encoder(self) -> Dict[NicoPPAttr, int]:
        return dict((NicoPPAttr(name), val) for name, val in self._get_label_mapping("subclass"))

    @cached_property
    def superclass_label_decoder(self) -> Dict[int, str]:
        return dict((val, name) for name, val in self._get_label_mapping("superclass"))

    @cached_property
    def subclass_label_decoder(self) -> Dict[int, str]:
        return dict((val, name) for name, val in self._get_label_mapping("subclass"))

    def _get_label_mapping(self, level: Literal["superclass", "subclass"]) -> List[Tuple[str, int]]:
        """Get a list of all possible (name, numerical value) pairs."""
        return list(
            self.metadata[[level, f"{level}_le"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

    def _check_unzipped(self) -> bool:
        return all((self._base_dir / attr).exists() for attr in NicoPPAttr)

    def _extract_metadata(self) -> None:
        """Extract concept/context/superclass information from the image filepaths and it save to csv."""
        self.logger.info("Extracting metadata.")
        image_paths: List[Path] = []
        image_paths.extend(self._base_dir.glob(f"**/*.jpg"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str)
        metadata = filepaths.str.split("/", expand=True).rename(
            columns={0: "subclass", 1: "superclass", 2: "filename"}
        )

        metadata["filepath"] = filepaths
        metadata.sort_index(axis=1, inplace=True)
        metadata.sort_values(by=["filepath"], axis=0, inplace=True)
        metadata = self._label_encode_metadata(metadata)
        metadata.to_csv(self._metadata_path, index=False)

    @staticmethod
    def _label_encode_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
        """Label encode the extracted concept/context/superclass information."""
        for col in metadata.columns:
            # Skip over filepath and filename columns
            if "file" not in col:
                # Add a new column containing the label-encoded data
                metadata[f"{col}_le"] = metadata[col].factorize()[0]
        return metadata
