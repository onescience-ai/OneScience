"""MACE data compatibility layer backed by datapipes."""

from __future__ import annotations

from onescience.datapipes.materials.pyg_stack.core.atomic_data import (
    AtomicData,
    get_data_loader,
)
from onescience.datapipes.materials.pyg_stack.core.configuration import (
    Configurations,
    Configuration,
)
from onescience.datapipes.materials.pyg_stack.core.utils import (
    KeySpecification,
    config_from_atoms,
    config_from_atoms_list,
    load_from_xyz,
    update_keyspec_from_kwargs,
)
from onescience.datapipes.materials.pyg_stack.storage.hdf5_dataset import (
    HDF5Dataset,
    dataset_from_sharded_hdf5,
)
from onescience.datapipes.materials.pyg_stack.storage.lmdb_dataset import LMDBDataset
from onescience.datapipes.materials.pyg_stack.storage.text_dataset import TextDataset

__all__ = [
    "AtomicData",
    "Configuration",
    "Configurations",
    "HDF5Dataset",
    "KeySpecification",
    "LMDBDataset",
    "TextDataset",
    "config_from_atoms",
    "config_from_atoms_list",
    "dataset_from_sharded_hdf5",
    "get_data_loader",
    "load_from_xyz",
    "update_keyspec_from_kwargs",
]
