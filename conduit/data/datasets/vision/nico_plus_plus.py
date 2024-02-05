"""NICO Dataset."""

from enum import Enum, auto
from functools import cached_property
import json
from pathlib import Path
import random
from typing import ClassVar, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias

import pandas as pd
from ranzen import StrEnum
import torch
from torch import Tensor

from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["NICOPP", "NicoPPTarget", "NicoPPAttr", "NicoPPSplit"]


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


class NicoPPSplit(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


SampleType: TypeAlias = TernarySample


class NICOPP(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """NICO++ dataset."""

    SampleType: TypeAlias = TernarySample
    Superclass: TypeAlias = NicoPPTarget
    Subclass: TypeAlias = NicoPPAttr
    Split: TypeAlias = NicoPPSplit

    data_split_seed: ClassVar[int] = 666  # this is the seed from the paper
    num_samples_val_test: ClassVar[int] = 75  # this is the number from the paper
    subpath: ClassVar[Path] = Path("public_dg_0416") / "train"

    def __init__(
        self,
        root: Union[str, Path],
        *,
        transform: Optional[ImageTform] = None,
        superclasses: Optional[Sequence[Union[NicoPPTarget, str]]] = None,
        split: Optional[Union[NicoPPSplit, str]] = None,
    ) -> None:
        self.superclasses: Optional[List[NicoPPTarget]] = None
        if superclasses is not None:
            assert superclasses, "superclasses should be a non-empty list"
            self.superclasses = [NicoPPTarget(superclass) for superclass in superclasses]
        self.split = NicoPPSplit[split.upper()] if isinstance(split, str) else split

        self.root = Path(root)
        self._base_dir = self.root / "nico_plus_plus"
        self._metadata_path = self._base_dir / "metadata.csv"

        if not self._check_unzipped():
            raise FileNotFoundError(
                f"Data not found at location {self._base_dir.resolve()}. Have you downloaded it?"
            )
        if not self._metadata_path.exists():
            self._extract_metadata()

        self.metadata = pd.read_csv(self._metadata_path)

        if self.superclasses is not None:
            self.metadata = self.metadata[self.metadata["superclass"].isin(self.superclasses)]

        if self.split is not None:
            self.metadata = self.metadata[self.metadata["split"] == self.split.value]

        # Divide up the dataframe into its constituent arrays because indexing with pandas is
        # substantially slower than indexing with numpy/torch
        x = self.metadata["filename"].to_numpy()
        y = torch.as_tensor(self.metadata["y"].to_numpy(), dtype=torch.long)
        s = torch.as_tensor(self.metadata["a"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, s=s, transform=transform, image_dir=self._base_dir)

    @cached_property
    def class_tree(self) -> Dict[str, Set[str]]:
        return self.metadata[["y", "a"]].drop_duplicates().groupby("y").agg(set).to_dict()["a"]

    @cached_property
    def superclass_label_decoder(self) -> Dict[int, NicoPPTarget]:
        return dict((val, NicoPPTarget(name)) for name, val in self._get_label_mapping("y"))

    @cached_property
    def subclass_label_decoder(self) -> Dict[int, NicoPPAttr]:
        return dict((val, NicoPPAttr(name)) for name, val in self._get_label_mapping("a"))

    def _get_label_mapping(self, level: Literal["y", "a"]) -> List[Tuple[str, int]]:
        """Get a list of all possible (name, numerical value) pairs."""
        return [
            (name, num)
            for num, name in self.metadata[[f"{level}_name", level]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ]

    def _check_unzipped(self) -> bool:
        if not all((self._base_dir / self.subpath / attr).exists() for attr in NicoPPAttr):
            return False
        if not (self._base_dir / "dg_label_id_mapping.json").exists():
            return False
        return True

    def _extract_metadata(self) -> None:
        self.logger.info("Generating metadata for NICO++...")
        attributes = ["autumn", "dim", "grass", "outdoor", "rock", "water"]  # 6 attrs, 60 labels
        meta = json.load((self._base_dir / "dg_label_id_mapping.json").open("r"))

        def _make_balanced_testset(
            df: pd.DataFrame, *, seed: int, num_samples_val_test: int
        ) -> pd.DataFrame:
            # each group has a test set size of (2/3 * num_samples_val_test) and a val set size of
            # (1/3 * num_samples_val_test); if total samples in original group < num_samples_val_test,
            # val/test will still be split by 1:2, but no training samples remained

            random.seed(seed)
            val_set, test_set = [], []
            for g in pd.unique(df["g"]):
                df_group = df[df["g"] == g]
                curr_data = df_group["filename"].values
                random.shuffle(curr_data)
                split_size = min(len(curr_data), num_samples_val_test)
                val_set += list(curr_data[: split_size // 3])
                test_set += list(curr_data[split_size // 3 : split_size])
            self.logger.info(f"Val: {len(val_set)}, Test: {len(test_set)}")
            assert len(set(val_set).intersection(set(test_set))) == 0
            combined_set = dict(zip(val_set, [NicoPPSplit.VAL.value for _ in range(len(val_set))]))
            combined_set.update(
                dict(zip(test_set, [NicoPPSplit.TEST.value for _ in range(len(test_set))]))
            )
            df["split"] = df["filename"].map(combined_set)
            df["split"].fillna(NicoPPSplit.TRAIN.value, inplace=True)
            df["split"] = df.split.astype(int)
            return df

        all_data = []
        for c, attr in enumerate(attributes):
            for label in meta:
                folder_path = self._base_dir / self.subpath / attr / label
                y = meta[label]
                for img_path in Path(folder_path).glob("*.jpg"):
                    all_data.append(
                        {
                            "filename": str(img_path.relative_to(self._base_dir)),
                            "y": y,
                            "a": c,
                            "y_name": label,
                            "a_name": attr,
                        }
                    )
        df = pd.DataFrame(all_data)
        df["g"] = df["a"] + df["y"] * len(attributes)
        df = _make_balanced_testset(
            df, seed=self.data_split_seed, num_samples_val_test=self.num_samples_val_test
        )
        df = df.drop(columns=["g"])
        df.to_csv(self._metadata_path, index=False)
