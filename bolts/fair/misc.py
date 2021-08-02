from enum import Enum, auto

__all__ = ["FairnessType"]


class FairnessType(Enum):
    DP = auto()
    EO = auto()
    EqOp = auto()
    No = auto()
