from __future__ import annotations
from enum import Enum, auto
from typing import TypeVar

from typing_extensions import Literal

__all__ = ["Stage", "str_to_enum", "TrainingMode"]


Stage = Literal["fit", "validate", "test"]


class TrainingMode(Enum):
    epoch = auto()
    step = auto()


T_co = TypeVar("T_co", covariant=True, bound=Enum)


def str_to_enum(*, str_: str, enum: type[T_co]) -> T_co:
    try:
        return enum[str_]  # type: ignore
    except KeyError:
        valid_ls = [mem.name for mem in enum]
        raise TypeError(
            f"'{str_}' is not a valid option for enum '{enum.__name__}'; must be one of {valid_ls}."
        )
