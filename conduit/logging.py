from __future__ import annotations
import logging
import sys
from typing import Optional

__all__ = ["init_logger"]


def init_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(level)
    return logger
