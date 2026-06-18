

"""An implementation of the inference pipeline of AlphaFold 3."""
from importlib import resources
from pathlib import Path
import os
import warnings


def _data_artifacts_exist() -> bool:
    try:
        root = Path(
            resources.files(__name__ + ".constants.converters")
        )
    except Exception:
        return False
    return (root / "ccd.pickle").exists() and (root / "chemical_component_sets.pickle").exists()


if not _data_artifacts_exist():
    warnings.warn(
        "AlphaFold3 data files (ccd.pickle, chemical_component_sets.pickle) are missing.\n"
        "Please run the following command once to build them:\n\n"
        "    python -m onescience.flax_models.alphafold3.build_data\n",
        stacklevel=1,
    )
