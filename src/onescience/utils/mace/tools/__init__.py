"""MACE training/tooling utilities."""

from __future__ import annotations

from onescience.datapipes.materials.tools import scatter, torch_geometric, torch_tools, utils
from onescience.datapipes.materials.tools.keys import DefaultKeys
from onescience.datapipes.materials.tools.torch_tools import (
    TensorDict,
    cartesian_to_spherical,
    count_parameters,
    init_device,
    init_wandb,
    set_default_dtype,
    set_seeds,
    spherical_to_cartesian,
    to_numpy,
    to_one_hot,
    voigt_to_matrix,
)
from onescience.datapipes.materials.tools.utils import (
    AtomicNumberTable,
    MetricsLogger,
    atomic_numbers_to_indices,
    compute_c,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    get_atomic_number_table_from_zs,
    get_tag,
    setup_logger,
)

from .arg_parser import build_default_arg_parser, build_preprocess_arg_parser
from .arg_parser_tools import check_args
from .cg import U_matrix_real
from .checkpoint import CheckpointHandler, CheckpointIO, CheckpointState
from .finetuning_utils import load_foundations, load_foundations_elements
from .train import SWAContainer, evaluate, train

__all__ = [
    "AtomicNumberTable",
    "CheckpointHandler",
    "CheckpointIO",
    "CheckpointState",
    "DefaultKeys",
    "MetricsLogger",
    "SWAContainer",
    "TensorDict",
    "U_matrix_real",
    "atomic_numbers_to_indices",
    "build_default_arg_parser",
    "build_preprocess_arg_parser",
    "cartesian_to_spherical",
    "check_args",
    "compute_c",
    "compute_mae",
    "compute_q95",
    "compute_rel_mae",
    "compute_rel_rmse",
    "compute_rmse",
    "count_parameters",
    "evaluate",
    "get_atomic_number_table_from_zs",
    "get_tag",
    "init_device",
    "init_wandb",
    "load_foundations",
    "load_foundations_elements",
    "scatter",
    "set_default_dtype",
    "set_seeds",
    "setup_logger",
    "spherical_to_cartesian",
    "to_numpy",
    "to_one_hot",
    "torch_geometric",
    "torch_tools",
    "train",
    "utils",
    "voigt_to_matrix",
]
