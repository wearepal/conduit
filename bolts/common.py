from __future__ import annotations
import collections
from typing import Generic

from typing_extensions import ClassVar, Literal

__all__ = ["Stage"]

from typing_inspect import typingGenericAlias

Stage = Literal["fit", "validate", "test"]


def get_args(tp):
    """Get type arguments."""
    # Note special aliases on Python 3.9 don't have __args__.
    if isinstance(tp, typingGenericAlias) and hasattr(tp, '__args__'):
        res = tp.__args__
        if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
            res = (list(res[:-1]), res[-1])
        return res
    return ()


def get_origin(tp):
    """Get the unsubscripted version of a type."""
    if isinstance(tp, typingGenericAlias):
        return tp.__origin__ if tp.__origin__ is not ClassVar else None
    if tp is Generic:
        return Generic
    return None
