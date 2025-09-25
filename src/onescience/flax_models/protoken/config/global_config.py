"""Global config"""

import copy

import ml_collections

# Global config
GLOBAL_CONFIG = {
    "sharding": True,
    "bf16_flag": True,
    "jax_small": 1e-5,
    "norm_small": 1e-5,
    "remat_flag": True,
    "use_dropout": False,  # True
    "global_dropout_rate": 0.0,
}
GLOBAL_CONFIG = ml_collections.ConfigDict(copy.deepcopy(GLOBAL_CONFIG))
