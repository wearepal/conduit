import logging
import sys
from typing import Any, Optional

__all__ = ["LoggingContext", "init_logger"]


class LoggingContext:
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        handler: Optional[logging.Handler] = None,
        close: bool = True,
    ) -> None:
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self) -> None:
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def init_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(level)
    return logger
