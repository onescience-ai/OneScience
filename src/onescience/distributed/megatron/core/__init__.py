# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import onescience.distributed.megatron.core.tensor_parallel
import onescience.distributed.megatron.core.utils
from onescience.distributed.megatron.core import parallel_state
from onescience.distributed.megatron.core.distributed import DistributedDataParallel
from onescience.distributed.megatron.core.inference_params import InferenceParams
from onescience.distributed.megatron.core.model_parallel_config import ModelParallelConfig
from onescience.distributed.megatron.core.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)
from onescience.distributed.megatron.core.timers import Timers

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
    "DistributedDataParallel",
    "InferenceParams",
    "ModelParallelConfig",
    "Timers",
]
