from ml_collections.config_dict import ConfigDict

global_config = {
    "sparse_flag": False,
    "bf16_flag": True,
    "dropout_flag": False,
    "remat_flag": False,
    "test_flag": True,
    "norm_small": 1e-6,
    "norm_method": "layernorm",
}

global_config = ConfigDict(global_config)