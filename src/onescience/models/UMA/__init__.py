

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from onescience.models.UMA._config import clear_cache
from onescience.models.UMA.calculate import pretrained_mlip
from onescience.models.UMA.calculate.ase_calculator import FAIRChemCalculator

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""

__all__ = ["FAIRChemCalculator", "pretrained_mlip", "clear_cache"]
