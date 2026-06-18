"""Utilities and compatibility exports for MACE training/tooling."""

from __future__ import annotations

import importlib

_SUBMODULES = {"calculators", "cli", "data", "modules", "tools"}
__all__ = sorted(_SUBMODULES)

def __getattr__(name: str):
    if name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))

