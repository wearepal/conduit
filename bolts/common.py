from __future__ import annotations
from enum import Enum, auto

from typing_extensions import Literal

__all__ = ["Stage"]


Stage = Literal["fit", "validate", "test"]


class FairnessType(Enum):
    DP = auto()
    EO = auto()
    EqOp = auto()
    No = auto()
