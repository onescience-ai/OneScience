import os
import re
import click
from pathlib import Path
from ..core.runner import PROJECT_ROOT


CONFIG_FILE = PROJECT_ROOT / ".env"


def _read_config() -> str:
    """安全读取 .env 文件内容"""
    if not CONFIG_FILE.exists():
        return ""
    try:
        return CONFIG_FILE.read_text(encoding="utf-8")
    except (OSError, PermissionError) as e:
        click.secho(f"读取配置文件失败: {e}", fg="red", err=True)
        return ""


def _write_config(content: str) -> bool:
    """安全写入 .env 文件"""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(content, encoding="utf-8")
        return True
    except (OSError, PermissionError) as e:
        click.secho(f"写入配置文件失败: {e}", fg="red", err=True)
        return False


@click.group("config")
def config_group():
    """查看或修改配置"""


@config_group.command("show")
def show_config():
    """查看当前配置"""
    click.secho("OneScience 配置", fg="green")
    if CONFIG_FILE.exists():
        click.echo(f"配置文件: {CONFIG_FILE}")
        config_text = _read_config()
        if not config_text:
            return
        for line in config_text.splitlines():
            if line.startswith("export "):
                parts = line[7:].split("=", 1)
                key = parts[0]
                val = parts[1].strip("\"'") if len(parts) > 1 else ""
                click.echo(f"  {key:<30} {val}")
    else:
        click.echo("配置文件不存在")
    click.echo("环境变量:")
    for key in ["ONESCIENCE_DATASETS_DIR", "ONESCIENCE_MODELS_DIR",
                "device", "num_nodes", "gpus_per_node", "distributed_backend"]:
        val = os.environ.get(key)
        if val:
            click.echo(f"  {key:<30} {val}")
    click.echo(f"  项目根目录: {PROJECT_ROOT}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def set_config(key, value):
    """修改配置项

    支持的键:

    路径配置:
      data_dir / dataset_dir / data_path  → ONESCIENCE_DATASETS_DIR
      model_dir / model_path              → ONESCIENCE_MODELS_DIR

    设备配置:
      device                              → 运行设备 (gpu/dcu/cpu)

    分布式配置:
      num_nodes                           → 节点数量
      gpus                                → 每节点 GPU 数量
      distributed_backend                 → 分布式后端 (nccl/gloo/mpi)
    """
    key_map = {
        "data_dir": "ONESCIENCE_DATASETS_DIR",
        "dataset_dir": "ONESCIENCE_DATASETS_DIR",
        "data_path": "ONESCIENCE_DATASETS_DIR",
        "model_dir": "ONESCIENCE_MODELS_DIR",
        "model_path": "ONESCIENCE_MODELS_DIR",
        "device": "device",
        "num_nodes": "num_nodes",
        "gpus": "gpus_per_node",
        "gpus_per_node": "gpus_per_node",
        "distributed_backend": "distributed_backend",
        "backend": "distributed_backend",
    }
    env_key = key_map.get(key)
    if not env_key:
        click.secho(f"不支持配置项: {key}", fg="red")
        click.echo(f"支持的配置项: {', '.join(key_map.keys())}")
        return

    existing = _read_config()
    pattern = re.compile(rf"^export {re.escape(env_key)}=.*", re.MULTILINE)
    if pattern.search(existing):
        existing = pattern.sub(f'export {env_key}="{value}"', existing)
    else:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        existing += f'export {env_key}="{value}"\n'

    if _write_config(existing):
        os.environ[env_key] = value
        click.secho(f"已设置 {env_key}={value}", fg="green")
