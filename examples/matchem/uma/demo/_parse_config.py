#!/usr/bin/env python3
"""UMA demo YAML 配置解析器 - 把 YAML 转换为训练启动命令与相关变量。

用法:
    python _parse_config.py <config.yaml> <action> [hydra_config_path]

action:
    name           打印实验名称
    command        打印训练启动命令 (python/torchrun/srun 风格)
                   需要额外传入 hydra_config_path，作为 train.py -c 的目标
    env            打印环境变量 export 语句
    env-args       打印 env_setup.sh 的参数 (conda_env module1 module2 ...)
    data-files     打印 preflight_check.sh 要检查的路径
                   (checkpoint_location + data.train_dataset.splits.*.src + data.val_dataset.splits.*.src)
    slurm          打印 SLURM 配置变量
    hydra-config   把 YAML 剥离 demo meta 后的部分(纯 hydra 内容)打印到 stdout
                   run.sh 会把它重定向到 outputs/xxx_ts/hydra_config.yaml

顶层 meta 键（只给 demo 工具用，不进 hydra）：
    name, description, launch, env, slurm, nccl

其余顶层键（data, job, runner, reducer, train_dataset, ...）作为 hydra 内容。
"""

import os
import sys

import yaml


# 这些顶层键只给 demo 工具用，在写 hydra_config 时会被剥离
META_KEYS = {"name", "description", "launch", "env", "slurm", "nccl"}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_train_py_path(config_path):
    """根据 demo/configs/xxx.yaml 的位置推算 UMA/train.py 路径。"""
    # config_path: .../UMA/demo/configs/xxx.yaml
    demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    uma_dir = os.path.dirname(demo_dir)
    return os.path.join(uma_dir, "train.py")


# -----------------------------------------------------------------------------
# action 实现
# -----------------------------------------------------------------------------

def print_name(cfg):
    print(cfg.get("name", "uma_train"))


def print_command(cfg, config_path, hydra_config_path):
    """拼 python/torchrun/srun train.py -c <hydra_config_path>。"""
    if not hydra_config_path:
        print(
            "[ERROR] command 动作需要提供 hydra_config_path 作为第三个参数",
            file=sys.stderr,
        )
        sys.exit(1)

    launch = cfg.get("launch", {}) or {}
    num_nodes = launch.get("num_nodes", 1)
    num_gpus = launch.get("num_gpus", 1)
    launcher = launch.get("launcher", "python")

    train_py = get_train_py_path(config_path)

    if num_nodes > 1:
        # 多节点: sbatch 内会用 srun 包裹 torchrun, 命令本体是 torchrun + train.py
        # 注意: master_addr/master_port/node_rank 在 run.sh 生成的 submit.sh 里
        # 基于 $SLURM_* 变量导出, 这里占位用 ${MASTER_ADDR} 等 shell 变量
        print(
            "torchrun \\\n"
            f"  --nnodes={num_nodes} \\\n"
            "  --node_rank=${SLURM_NODEID} \\\n"
            f"  --nproc_per_node={num_gpus} \\\n"
            "  --rdzv_id=${SLURM_JOB_ID} \\\n"
            "  --rdzv_backend=c10d \\\n"
            "  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \\\n"
            f"  {train_py} \\\n"
            f"  -c {hydra_config_path}"
        )
    elif launcher == "torchrun" or num_gpus > 1:
        # 单节点多卡: torchrun
        print(
            "torchrun \\\n"
            "  --nnodes=1 \\\n"
            f"  --nproc_per_node={num_gpus} \\\n"
            f"  {train_py} \\\n"
            f"  -c {hydra_config_path}"
        )
    else:
        # 单卡: 直接 python
        print(f"python {train_py} -c {hydra_config_path}")


def print_env(cfg):
    """打印环境变量 export 语句。"""
    launch = cfg.get("launch", {}) or {}
    num_nodes = launch.get("num_nodes", 1)
    num_gpus = launch.get("num_gpus", 1)

    omp = launch.get("omp_num_threads", 1)
    print(f"export OMP_NUM_THREADS={omp}")

    if num_gpus > 1:
        devices = ",".join(str(i) for i in range(num_gpus))
        print(f"export HIP_VISIBLE_DEVICES={devices}")
    elif num_gpus == 1:
        print("export HIP_VISIBLE_DEVICES=0")

    # 多节点 NCCL 配置
    if num_nodes > 1:
        nccl = cfg.get("nccl", {}) or {}
        print("export HSA_FORCE_FINE_GRAIN_PCIE=1")
        if nccl.get("socket_ifname"):
            print(f"export NCCL_SOCKET_IFNAME={nccl['socket_ifname']}")
        if nccl.get("ib_hca"):
            print(f"export NCCL_IB_HCA={nccl['ib_hca']}")
        if nccl.get("proto"):
            print(f"export NCCL_PROTO={nccl['proto']}")


def print_env_args(cfg):
    """打印 env_setup.sh 的参数: conda_env module1 module2 ..."""
    env = cfg.get("env", {}) or {}
    conda_env = env.get("conda_env", "chem")
    modules = env.get("modules", []) or []
    parts = [conda_env] + list(modules)
    print(" ".join(parts))


def _dig_src_paths(node, out):
    """递归找出 splits 下所有 src 字段。"""
    if isinstance(node, dict):
        if "splits" in node and isinstance(node["splits"], dict):
            for split_name, split_val in node["splits"].items():
                if isinstance(split_val, dict) and "src" in split_val:
                    out.append(split_val["src"])
        for v in node.values():
            _dig_src_paths(v, out)
    elif isinstance(node, list):
        for item in node:
            _dig_src_paths(item, out)


def _dig_checkpoint_locations(node, out):
    """递归找出 checkpoint_location 字段。"""
    if isinstance(node, dict):
        if "checkpoint_location" in node and isinstance(
            node["checkpoint_location"], str
        ):
            out.append(node["checkpoint_location"])
        for v in node.values():
            _dig_checkpoint_locations(v, out)
    elif isinstance(node, list):
        for item in node:
            _dig_checkpoint_locations(item, out)


def print_data_files(cfg):
    """打印 preflight 要检查的路径: checkpoint + 数据 src。"""
    paths = []
    data = cfg.get("data", {})
    _dig_src_paths(data, paths)
    _dig_checkpoint_locations(cfg, paths)
    # 去重保持顺序
    seen = set()
    for p in paths:
        if p and p not in seen and not str(p).startswith("??"):
            seen.add(p)
            print(p)


def print_slurm(cfg):
    """打印 SLURM 配置变量。"""
    launch = cfg.get("launch", {}) or {}
    slurm = cfg.get("slurm", {}) or {}
    num_nodes = launch.get("num_nodes", 1)
    num_gpus = launch.get("num_gpus", 1)

    name = cfg.get("name", "uma_train")
    partition = slurm.get("partition", "k100ai")
    time_limit = slurm.get("time", "12:00:00")
    cpus = slurm.get("cpus_per_task", 128)

    # UMA 的多节点策略: 每个节点 1 个 srun task, 由 torchrun 在该节点内再 spawn
    # nproc_per_node=num_gpus 个进程. 因此无论单/多节点 NTASKS_PER_NODE 都是 1.
    ntasks = 1
    cpus_per_task = cpus
    _ = num_gpus  # 保留 num_gpus 以便 GPUS_PER_NODE 正确填写

    print(f"JOB_NAME={name}")
    print(f"PARTITION={partition}")
    print(f"NODES={num_nodes}")
    print(f"NTASKS_PER_NODE={ntasks}")
    print(f"CPUS_PER_TASK={cpus_per_task}")
    print(f"GPUS_PER_NODE={num_gpus}")
    print(f"TIME={time_limit}")


def print_hydra_config(cfg):
    """把 cfg 剥离 META_KEYS 后的部分用 YAML 输出。"""
    hydra_only = {k: v for k, v in cfg.items() if k not in META_KEYS}
    # sort_keys=False 保留原顺序; default_flow_style=False 强制 block 风格
    yaml.safe_dump(
        hydra_only,
        sys.stdout,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    action = sys.argv[2]
    hydra_config_path = sys.argv[3] if len(sys.argv) >= 4 else None

    cfg = load_config(config_path)

    actions = {
        "name": lambda: print_name(cfg),
        "command": lambda: print_command(cfg, config_path, hydra_config_path),
        "env": lambda: print_env(cfg),
        "env-args": lambda: print_env_args(cfg),
        "data-files": lambda: print_data_files(cfg),
        "slurm": lambda: print_slurm(cfg),
        "hydra-config": lambda: print_hydra_config(cfg),
    }

    fn = actions.get(action)
    if fn is None:
        print(f"未知 action: {action}", file=sys.stderr)
        print(f"可用: {', '.join(actions.keys())}", file=sys.stderr)
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
