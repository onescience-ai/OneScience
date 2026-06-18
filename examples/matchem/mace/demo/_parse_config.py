#!/usr/bin/env python3
"""Parse MACE demo YAML and print derived values for run.sh.

Usage:
    python _parse_config.py <config.yaml> <action>

Actions:
    command      Print launch command
    env          Print environment exports
    env-args     Print arguments for env_setup.sh
    data-files   Print input files for preflight check
    slurm        Print SLURM variables
    name         Print experiment name
"""

from __future__ import annotations

import os
import shlex
import sys
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _needs_quoting(value: str) -> bool:
    special = set(" {}()[]|&;'\"\\!`<>*?~#")
    return any(c in special for c in value)


def _quote_value(value: Any) -> str:
    s = str(value)
    if "$" in s:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return shlex.quote(s)


def _build_cli_args(arg_map: dict[str, Any], auto_name: str | None = None) -> list[str]:
    args: list[str] = []
    if auto_name and "name" not in arg_map:
        args.append(f"--name={_quote_value(auto_name)}")

    for key, value in arg_map.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        if isinstance(value, (dict, list)):
            args.append(f"--{key}={_quote_value(value)}")
            continue

        s = str(value)
        if _needs_quoting(s):
            args.append(f"--{key}={_quote_value(s)}")
        else:
            args.append(f"--{key}={s}")
    return args


def _format_command_with_args(prefix: str, args: list[str]) -> str:
    if not args:
        return prefix
    return f"{prefix} \
  " + " \
  ".join(args)


def _demo_dir(config_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(config_path)))


def _mace_dir(config_path: str) -> str:
    return os.path.dirname(_demo_dir(config_path))


def _get_train_py_path(config_path: str) -> str:
    return os.path.join(_mace_dir(config_path), "train.py")


def _resolve_script_path(config_path: str, script_value: str) -> str:
    if os.path.isabs(script_value):
        return script_value

    config_dir = os.path.dirname(os.path.abspath(config_path))
    demo_dir = _demo_dir(config_path)
    mace_dir = _mace_dir(config_path)

    candidates = [
        os.path.join(config_dir, script_value),
        os.path.join(demo_dir, script_value),
        os.path.join(mace_dir, script_value),
        os.path.join(mace_dir, "scripts", script_value),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Return the most likely path even if it does not exist; preflight will catch it.
    return os.path.join(mace_dir, "scripts", script_value)


def _is_entrypoint_mode(cfg: dict[str, Any]) -> bool:
    entrypoint = cfg.get("entrypoint")
    return isinstance(entrypoint, dict) and bool(entrypoint.get("script"))


def _build_train_command(cfg: dict[str, Any], config_path: str) -> str:
    launch = cfg.get("launch", {}) or {}
    num_nodes = int(launch.get("num_nodes", 1))
    num_gpus = int(launch.get("num_gpus", 1))
    launcher = launch.get("launcher", "python")

    train_py = _get_train_py_path(config_path)
    train_args = _build_cli_args(
        cfg.get("train_args", {}) or {},
        auto_name=cfg.get("name"),
    )

    if num_nodes > 1 and "--distributed" not in train_args:
        train_args.append("--distributed")
    if launcher == "torchrun" and "--distributed" not in train_args:
        train_args.append("--distributed")

    if num_nodes > 1:
        return _format_command_with_args(f"python {train_py}", train_args)
    if launcher == "torchrun":
        prefix = (
            "torchrun \
  --nnodes=1 \
  "
            f"--nproc_per_node={num_gpus} \
  {train_py}"
        )
        return _format_command_with_args(prefix, train_args)
    return _format_command_with_args(f"python {train_py}", train_args)


def _build_entrypoint_command(cfg: dict[str, Any], config_path: str) -> str:
    entrypoint = cfg.get("entrypoint", {}) or {}
    script_path = _resolve_script_path(config_path, str(entrypoint["script"]))
    args = _build_cli_args(entrypoint.get("args", {}) or {})
    return _format_command_with_args(f"python {script_path}", args)


def print_command(cfg: dict[str, Any], config_path: str) -> None:
    if _is_entrypoint_mode(cfg):
        print(_build_entrypoint_command(cfg, config_path))
    else:
        print(_build_train_command(cfg, config_path))


def print_env(cfg: dict[str, Any]) -> None:
    launch = cfg.get("launch", {}) or {}
    num_nodes = int(launch.get("num_nodes", 1))
    num_gpus = int(launch.get("num_gpus", 1))

    omp = int(launch.get("omp_num_threads", 1))
    print(f"export OMP_NUM_THREADS={omp}")

    if num_gpus > 1:
        devices = ",".join(str(i) for i in range(num_gpus))
        print(f"export HIP_VISIBLE_DEVICES={devices}")
    elif num_gpus == 1:
        print("export HIP_VISIBLE_DEVICES=0")

    if num_nodes > 1:
        nccl = cfg.get("nccl", {}) or {}
        print("export HSA_FORCE_FINE_GRAIN_PCIE=1")
        if nccl.get("socket_ifname"):
            print(f"export NCCL_SOCKET_IFNAME={nccl['socket_ifname']}")
        if nccl.get("ib_hca"):
            print(f"export NCCL_IB_HCA={nccl['ib_hca']}")
        if nccl.get("proto"):
            print(f"export NCCL_PROTO={nccl['proto']}")

    # 用户自定义额外环境变量（可覆盖上面自动生成的变量）
    extra_env = cfg.get("extra_env", {})
    for key, value in extra_env.items():
        if value is None or value == "":
            # 空值时只导出 key，保留当前 shell 中的值
            print(f"export {key}")
        else:
            print(f"export {key}={_quote_value(str(value))}")


def print_env_args(cfg: dict[str, Any]) -> None:
    env = cfg.get("env", {}) or {}
    conda_env = env.get("conda_env", "chem")
    modules = env.get("modules", []) or []
    parts = [str(conda_env)] + [str(m) for m in modules]
    print(" ".join(parts))


def _print_unique(paths: list[str]) -> None:
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        print(path)


def print_data_files(cfg: dict[str, Any], config_path: str) -> None:
    explicit = cfg.get("preflight_files")
    if isinstance(explicit, list):
        _print_unique([str(x) for x in explicit])
        return

    if _is_entrypoint_mode(cfg):
        entrypoint = cfg.get("entrypoint", {}) or {}
        files: list[str] = []
        data_files = entrypoint.get("data_files", [])
        if isinstance(data_files, list):
            files.extend(str(x) for x in data_files)
        script_value = entrypoint.get("script")
        if script_value:
            files.append(_resolve_script_path(config_path, str(script_value)))
        _print_unique(files)
        return

    train_args = cfg.get("train_args", {}) or {}
    files: list[str] = []
    for key in (
        "train_file",
        "valid_file",
        "test_file",
        "statistics_file",
        "foundation_model",
        "pt_train_file",
    ):
        val = train_args.get(key)
        if val:
            files.append(str(val))
    _print_unique(files)


def print_slurm(cfg: dict[str, Any]) -> None:
    launch = cfg.get("launch", {}) or {}
    slurm = cfg.get("slurm", {}) or {}
    num_nodes = int(launch.get("num_nodes", 1))
    num_gpus = int(launch.get("num_gpus", 1))

    name = cfg.get("name", "mace_train")
    partition = slurm.get("partition", "newlarge")
    time_limit = slurm.get("time", "8:00:00")
    cpus = int(slurm.get("cpus_per_task", 64))

    if num_nodes > 1:
        ntasks = num_gpus
        cpus_per_task = max(1, cpus // max(1, num_gpus))
    else:
        ntasks = 1
        cpus_per_task = cpus

    print(f"JOB_NAME={name}")
    print(f"PARTITION={partition}")
    print(f"NODES={num_nodes}")
    print(f"NTASKS_PER_NODE={ntasks}")
    print(f"CPUS_PER_TASK={cpus_per_task}")
    print(f"GPUS_PER_NODE={num_gpus}")
    print(f"TIME={time_limit}")


def print_name(cfg: dict[str, Any]) -> None:
    print(cfg.get("name", "mace_train"))


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    action = sys.argv[2]
    cfg = load_config(config_path)

    actions = {
        "command": lambda: print_command(cfg, config_path),
        "env": lambda: print_env(cfg),
        "env-args": lambda: print_env_args(cfg),
        "data-files": lambda: print_data_files(cfg, config_path),
        "slurm": lambda: print_slurm(cfg),
        "name": lambda: print_name(cfg),
    }

    fn = actions.get(action)
    if fn is None:
        print(f"Unknown action: {action}", file=sys.stderr)
        print(f"Available: {', '.join(actions.keys())}", file=sys.stderr)
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
