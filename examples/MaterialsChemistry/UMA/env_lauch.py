# env_launch.py
from __future__ import annotations
import argparse
import logging
import os
import random
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError
from torch.profiler import profile, ProfilerActivity
from onescience.models.UMA.common import gp_utils  # noqa: F401
#from onescience.models.UMA.common import distutils
from onescience.distributed.manager import DistributedManager
from onescience.models.UMA.common.logger import WandBSingletonLogger
from onescience.models.UMA.common.utils import (
    get_commit_hash,
    get_timestamp_uid,
    setup_env_vars,
    setup_logging,
)

# ---- 基本日志 ----
logging.basicConfig(level=logging.INFO)

ALLOWED_TOP_LEVEL_KEYS = {"job", "runner", "reducer"}
LOG_DIR_NAME = "logs"
CHECKPOINT_DIR_NAME = "checkpoints"
RESULTS_DIR = "results"
CONFIG_FILE_NAME = "canonical_config.yaml"

# ------- JobConfig（保留最小字段；YAML 里的 scheduler 会被忽略） -------
from dataclasses import dataclass, field
from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class Metadata:
    commit: str
    log_dir: str
    checkpoint_dir: str
    results_dir: str
    config_path: str


@dataclass
class JobConfig:
    run_name: str = field(default_factory=lambda: get_timestamp_uid())
    timestamp_id: str = field(default_factory=lambda: get_timestamp_uid())
    run_dir: str = "/tmp"  # 建议在 YAML 中设置
    device_type: DeviceType = DeviceType.CUDA
    debug: bool = False
    logger: Optional[dict] = None
    seed: int = 0
    deterministic: bool = False
    runner_state_path: Optional[str] = None
    metadata: Optional[Metadata] = None
    graph_parallel_group_size: Optional[int] = None
    scheduler: Optional[Any] = None

    def __post_init__(self) -> None:
        self.run_dir = os.path.abspath(self.run_dir)
        self.metadata = Metadata(
            commit=get_commit_hash(),
            log_dir=os.path.join(self.run_dir, self.timestamp_id, LOG_DIR_NAME),
            checkpoint_dir=os.path.join(
                self.run_dir, self.timestamp_id, CHECKPOINT_DIR_NAME
            ),
            results_dir=os.path.join(self.run_dir, self.timestamp_id, RESULTS_DIR),
            config_path=os.path.join(self.run_dir, self.timestamp_id, CONFIG_FILE_NAME),
        )


# ----------------- 小工具 -----------------
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_deterministic_mode() -> None:
    logging.info("Setting deterministic mode!")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_canonical_config(config):
    # 初始化 metadata，因为 OmegaConf 对 dataclasses 不会自动调用 __post_init__
    job = OmegaConf.to_object(config.job)
    job.__post_init__()
    config.job = job

    # 清理未使用的顶层 keys（除了 ALLOWED_TOP_LEVEL_KEYS）
    all_keys = set(config.keys()).difference(ALLOWED_TOP_LEVEL_KEYS)
    used_keys = set()
    for key in all_keys:
        copy_cfg = OmegaConf.create({k: v for k, v in config.items() if k != key})
        try:
            OmegaConf.resolve(copy_cfg)
        except InterpolationKeyError:
            used_keys.add(key)
    unused_keys = all_keys.difference(used_keys)
    if unused_keys:
        raise ValueError(
            f"Found unused top-level keys in config: {unused_keys}. "
            f"Only {ALLOWED_TOP_LEVEL_KEYS} or used-as-variables are allowed."
        )

    OmegaConf.resolve(config)
    return OmegaConf.create(
        {k: v for k, v in config.items() if k in ALLOWED_TOP_LEVEL_KEYS}
    )


def get_hydra_config_from_yaml(config_yml: str, overrides_args: list[str]):
    import os, hydra
    from omegaconf import OmegaConf

    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory, version_base="1.1")
    cfg = hydra.compose(config_name=config_name, overrides=overrides_args or [])

    # 合入最小 JobConfig 结构
    cfg = OmegaConf.merge({"job": OmegaConf.structured(JobConfig)}, cfg)

    # >>> 这里强制所有 rank 共享一个 timestamp_id <<<
    shared_id = (
        os.environ.get("RUN_ID")  # 你在 sbatch 里自定义的
        or os.environ.get("SLURM_JOB_ID")  # Slurm 固定
        or os.environ.get("RDZV_ID")  # torchrun --rdzv_id
        or os.environ.get("TORCHELASTIC_RUN_ID")  # 某些版本的变量名
    )
    if shared_id:
        cfg.job.timestamp_id = str(shared_id)

    # 现在再做 canonicalize（会调用 __post_init__，据此生成统一的 metadata 路径）
    return get_canonical_config(cfg)

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config, override_args or [])

    # 目录与保存配置
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(cfg.job.metadata.log_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.job.metadata.config_path)
    logging.info(f"saved canonical config to {cfg.job.metadata.config_path}")

    # 环境与日志
    setup_env_vars()
    setup_logging()

    # <<< 核心修改：使用 DistributedManager 初始化分布式环境 >>>
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "gloo" if cfg.job.device_type == DeviceType.CPU else "nccl"
    
    dist_config = {
        "world_size": world_size,
        "distributed_backend": backend,
        "submit": False,  # 绝不走 Slurm/submitit 分支
        "cpu": (cfg.job.device_type == DeviceType.CPU),
        "timeout": 30,  # 可以根据需要设置合适的超时时间
    }

    logging.info(
        f"Init distributed from ENV: RANK={os.environ.get('RANK')}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={world_size}, "
        f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}"
    )

    # 通过 DistributedManager 初始化分布式环境
    DistributedManager.initialize()  # 使用现有的初始化方法

    # 配置图并行组（如果有的话）
    if cfg.job.graph_parallel_group_size is not None:
        gp_utils.setup_graph_parallel_groups(
            cfg.job.graph_parallel_group_size, dist_config["distributed_backend"]
        )

    # 设置随机性
    _set_seeds(cfg.job.seed)
    if cfg.job.deterministic:
        _set_deterministic_mode()

    # 日志系统（可选）
    if cfg.job.logger and DistributedManager().rank == 0 and not cfg.job.debug:
        logger_initializer = hydra.utils.instantiate(cfg.job.logger)
        simple_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        logger_initializer(
            config=simple_config,
            run_id=cfg.job.timestamp_id,
            run_name=cfg.job.run_name,
            log_dir=cfg.job.metadata.log_dir,
        )

    # 实例化并运行
    from onescience.models.UMA.components.reducer import Reducer  # noqa: F401
    from onescience.models.UMA.components.runner import Runner  # noqa: F401

    runner = hydra.utils.instantiate(cfg.runner)
    runner.job_config = cfg.job
    runner.load_state(cfg.job.runner_state_path)
    runner.run()

    DistributedManager.cleanup()  # 清理分布式资源


if __name__ == "__main__":
    main()