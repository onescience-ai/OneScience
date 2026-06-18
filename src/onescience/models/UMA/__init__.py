

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from onescience.models.UMA.base import HydraModel, HydraModelV2

try:
    __version__ = version("onescience.utils.uma")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""

__all__ = [
    "HydraModel",
    "HydraModelV2",
    "eSCNMDBackbone",
    "eSCNMDMoeBackbone",
]


def __getattr__(name: str):
    if name == "eSCNMDBackbone":
        from onescience.models.UMA.uma_escn_md import eSCNMDBackbone
        return eSCNMDBackbone
    if name == "eSCNMDMoeBackbone":
        from onescience.models.UMA.uma_escn_moe import eSCNMDMoeBackbone
        return eSCNMDMoeBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
