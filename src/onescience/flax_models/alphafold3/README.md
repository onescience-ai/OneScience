# AlphaFold 3 Integration for OneScience

This is the AlphaFold 3 implementation integrated into the OneScience framework as a submodule.

## Overview

AlphaFold 3 is a state-of-the-art machine learning model for predicting protein structure, developed by DeepMind. This submodule integrates AlphaFold 3 into the OneScience framework, allowing it to be used as part of larger scientific computing workflows.

## Installation

### As Part of OneScience

The recommended way to install alphafold3 is as part of the complete OneScience package:

```bash
# Install OneScience with alphafold3 support

# install jackhmmer
mkdir ~/hmmer_build ~/hmmer
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix ~/hmmer_build
cd ~/hmmer_build &&  tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz
patch -p0 < jackhmmer_seq_limit.patch
cd ~/hmmer-3.4
./configure --prefix ~/hmmer
make -j && make install && cd ./easel && make install
rm -R ~/hmmer_build

# install extension
pip install .[bio] -c constraints.txt
cp -r /public/onestore/onedatasets/alphafold3/_dep xxx/
export ALPHAFOLD3_DEP_DIR=/public/onestore/onedatasets/alphafold3/_dep
cd src/onescience/flax_models/alphafold3/
python build_extension.py

# optional create mmseqs2 database (please contact ai4s@sugon.com for mmseqs2 program)
export mmfasta=/root/public_databases
cd /root/public_databases && mkdir mmseqsDB
export mmdb=/root/public_databases/mmseqsDB
export CUDA_VISIBLE_DEVICES=0
mmseqs createdb $mmfasta/bfd-first_non_consensus_sequences.fasta $mmdb/small_bfd_db --gpu 1 --threads 32 --createdb-mode 2
mmseqs createdb $mmfasta/mgy_clusters_2022_05.fa $mmdb/mgnify_db --gpu 1 --threads 32 --createdb-mode 2
mmseqs createdb $mmfasta/uniprot_all_2021_04.fa $mmdb/uniprot_cluster_annot_db --gpu 1 --threads 32 --createdb-mode 2
mmseqs createdb $mmfasta/uniref90_2022_05.fa $mmdb/uniref90_db --gpu 1 --threads 32 --createdb-mode 2
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
