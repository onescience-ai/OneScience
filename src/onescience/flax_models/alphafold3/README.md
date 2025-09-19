# AlphaFold 3 Integration for OneScience

This is the AlphaFold 3 implementation integrated into the OneScience framework as a submodule.

## Overview

AlphaFold 3 is a state-of-the-art machine learning model for predicting protein structure, developed by DeepMind. This submodule integrates AlphaFold 3 into the OneScience framework, allowing it to be used as part of larger scientific computing workflows.

## Installation

### As Part of OneScience

The recommended way to install alphafold3 is as part of the complete OneScience package:

```bash
# Install OneScience with alphafold3 support
pip install .
export ALPHAFOLD3_DEP_DIR=～/osmodels/alphafold3/_dep
python src/onescience/flax_models/alphafold3/build_extension.py
```

## Usage

```python
# Import alphafold3 as part of onescience
import onescience.flax_models.alphafold3 as af3

# Access alphafold3 components
from onescience.flax_models.alphafold3 import structure, model, data

# Use alphafold3 functionality
print(f"AlphaFold3 version: {af3.__version__}")
```

## Requirements

- Python 3.11+
- JAX with CUDA support (optional, for GPU acceleration)
- CMake 3.28+ (for building C++ extensions)
- Additional dependencies listed in pyproject.toml

## License

This code is licensed under CC BY-NC-SA 4.0. See the original AlphaFold 3 repository for more details on usage restrictions and licensing terms.

## Citation

If you use this code in your research, please cite the AlphaFold 3 paper:

```
Abramson, J., Adler, J., Dunger, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630, 493–500 (2024).
```

## Links

- [Original AlphaFold 3 Repository](https://github.com/google-deepmind/alphafold3)
- [AlphaFold 3 Paper](https://www.nature.com/articles/s41586-024-07487-w) 