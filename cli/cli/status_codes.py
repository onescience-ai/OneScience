from __future__ import annotations
from enum import IntEnum


class ExitStatus(IntEnum):
    SUCCESS = 0
    ERROR = 1
    ERROR_CTRL_C = 2
    ERROR_TIMEOUT = 3
    ERROR_TOO_MANY_REDIRECTS = 4
    UNKNOWN_MODEL = 10
    MODEL_DIR_MISSING = 11
    SCRIPT_NOT_FOUND = 12
