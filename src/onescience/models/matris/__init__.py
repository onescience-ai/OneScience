"""
MatRIS: Material Representation Learning with Interatomic Structure

A deep learning model for predicting material properties including energy, 
forces, stress, and magnetic moments.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)  # read from pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"

# Import main model
from onescience.models.matris.matris import MatRIS

# Import graph components
from onescience.datapipes.materials.matris import GraphConverter, RadiusGraph

# Import training components (optional, for training)
try:
    from onescience.utils.matris import MatrisTrainer, MatrisLoss, compute_metrics
except ImportError:
    # Training dependencies might not be installed
    pass

__all__ = [
    "MatRIS",
    "GraphConverter", 
    "RadiusGraph",
    "__version__",
]
