"""Runtime logging setup utilities for UMA jobs."""

from __future__ import annotations

import logging
import os
import sys


class SeverityLevelBetween(logging.Filter):
    """Filter records by inclusive lower bound and exclusive upper bound."""

    def __init__(self, min_level: int, max_level: int) -> None:
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record) -> bool:
        return self.min_level <= record.levelno < self.max_level


def setup_logging() -> None:
    """Configure root logging once with stdout/stderr severity split."""
    root = logging.getLogger()
    target_logging_level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper())
    root.setLevel(target_logging_level)
    if root.hasHandlers():
        return

    log_formatter = logging.Formatter(
        "%(asctime)s %(pathname)s:%(lineno)d: (%(levelname)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler_out = logging.StreamHandler(sys.stdout)
    handler_out.addFilter(SeverityLevelBetween(target_logging_level, logging.WARNING))
    handler_out.setFormatter(log_formatter)
    root.addHandler(handler_out)

    handler_err = logging.StreamHandler(sys.stderr)
    handler_err.setLevel(logging.WARNING)
    handler_err.setFormatter(log_formatter)
    root.addHandler(handler_err)
