from __future__ import annotations

from typing_extensions import Literal

__all__ = ["Stage"]

Stage = Literal["fit", "validate", "test"]
