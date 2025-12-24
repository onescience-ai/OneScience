# onescience/models/mace/tools/__init__.py
# (Refactored to import from datapipes/materials)

# -----------------------------------------------------------------
# 1. 导入 "模型训练" 相关的、保留在本地的工具 (相对导入)
# -----------------------------------------------------------------
from .arg_parser import build_default_arg_parser, build_preprocess_arg_parser
from .arg_parser_tools import check_args
from .cg import U_matrix_real
from .checkpoint import CheckpointHandler, CheckpointIO, CheckpointState
from .finetuning_utils import load_foundations, load_foundations_elements
from .train import SWAContainer, evaluate, train

# -----------------------------------------------------------------
# 2. ✨ 关键重构：导入 "通用数据" 相关的、已迁移到 L3 的工具
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# 3. 导出 __all__ 列表 (保持不变)
#    (这个列表现在会导出我们从本地和 L3 导入的所有工具)
# -----------------------------------------------------------------
__all__ = [
    "TensorDict",
    "AtomicNumberTable",
    "atomic_numbers_to_indices",
    "to_numpy",
    "to_one_hot",
    "build_default_arg_parser",
    "check_args",
    "DefaultKeys",
    "set_seeds",
    "init_device",
    "setup_logger",
    "get_tag",
    "count_parameters",
    "MetricsLogger",
    "get_atomic_number_table_from_zs",
    "train",
    "evaluate",
    "SWAContainer",
    "CheckpointHandler",
    "CheckpointIO",
    "CheckpointState",
    "set_default_dtype",
    "compute_mae",
    "compute_rel_mae",
    "compute_rmse",
    "compute_rel_rmse",
    "compute_q95",
    "compute_c",
    "U_matrix_real",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "voigt_to_matrix",
    "init_wandb",
    "load_foundations",
    "load_foundations_elements",
    "build_preprocess_arg_parser",
]