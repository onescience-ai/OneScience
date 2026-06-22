import os
import click
from pathlib import Path
from ..core.registry import model_registry, EXAMPLES_DIR
from ..core.runner import RESULTS_DIR


def _get_model_dir(info: dict) -> Path:
    """获取模型目录，优先用 info['model_dir']"""
    model_dir = info.get("model_dir")
    if model_dir:
        return Path(model_dir)
    return EXAMPLES_DIR / info["domain"] / info["model"]


def _find_wandb_runs(model_dir: Path):
    """扫描模型目录下的 wandb 运行记录"""
    wandb_dirs = []
    for root, dirs, files in os.walk(str(model_dir)):
        for d in dirs:
            if d.startswith("wandb") or d == "wandb":
                wandb_dirs.append(Path(root) / d)
    return wandb_dirs


def _find_lightning_logs(model_dir: Path):
    """扫描模型目录下的 lightning_logs"""
    logs_dir = model_dir / "lightning_logs"
    if logs_dir.exists():
        return sorted(logs_dir.iterdir())
    return []


@click.group("experiment")
def experiment_group():
    """实验管理（查看执行记录）"""


@experiment_group.command("list")
@click.argument("model_alias", required=False)
@click.option("-domain", default=None, help="按领域筛选")
@click.option("-format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
def list_experiments(model_alias, domain, fmt):
    """列出实验记录"""
    from ..core.formatter import Formatter

    if model_alias:
        info = model_registry.resolve(model_alias)
        if not info:
            click.secho(f"未知模型: {model_alias}", fg="red")
            return
        models = [info]
    else:
        models = model_registry.list_models(domain)

    rows = []
    for m in models:
        model_dir = EXAMPLES_DIR / m["domain"] / m["model"]
        alias = m["alias"]

        # 检查 saved results
        result_dir = RESULTS_DIR / alias
        has_results = result_dir.exists() and len(list(result_dir.iterdir())) > 0

        # 检查 wandb 记录
        wandb_runs = _find_wandb_runs(model_dir)
        has_wandb = len(wandb_runs) > 0

        # 检查 lightning_logs
        logs = _find_lightning_logs(model_dir)
        has_logs = len(logs) > 0

        # 检查日志文件
        log_files = list(model_dir.glob("*_execution.log"))
        has_logs = has_logs or len(log_files) > 0

        status = []
        if has_results:
            status.append("结果")
        if has_wandb:
            status.append("W&B")
        if has_logs:
            status.append("日志")

        status_str = ", ".join(status) if status else "无"
        rows.append([alias, m["model"], m["domain"], status_str])

    if not rows:
        click.echo("未找到实验记录")
        return

    headers = ["模型别名", "模型名称", "领域", "记录"]
    if fmt == "json":
        click.echo(Formatter.json(rows))
    elif fmt == "csv":
        click.echo(Formatter.csv(headers, rows))
    else:
        click.echo(Formatter.table(headers, rows))


@experiment_group.command("info")
@click.argument("model_alias")
def experiment_info(model_alias):
    """显示模型的详细实验记录"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return

    model_dir = _get_model_dir(info)
    alias = info["alias"]

    click.secho(f"实验详情: {alias}", fg="green")
    click.echo(f"  模型: {info['model']}")
    click.echo(f"  领域: {info['domain']}")
    click.echo(f"  目录: {model_dir}")
    click.echo("")

    # 已保存的结果
    result_dir = RESULTS_DIR / alias
    click.secho("已保存的结果:", fg="cyan")
    if result_dir.exists():
        for item in sorted(result_dir.iterdir()):
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                click.echo(f"  {item.name}/  ({_format_size(size)})")
            elif item.name == "metrics.json":
                click.echo(f"  metrics.json (指标数据)")
            else:
                click.echo(f"  {item.name}")
    else:
        click.echo("  （无）")

    # 日志文件
    click.secho("\n执行日志:", fg="cyan")
    log_files = sorted(model_dir.glob("*_execution.log"))
    if log_files:
        for lf in log_files:
            size = lf.stat().st_size if lf.exists() else 0
            click.echo(f"  {lf.name:40s} {_format_size(size)}")
    else:
        click.echo("  （无）")

    # Lightning logs
    logs_dir = model_dir / "lightning_logs"
    if logs_dir.exists():
        versions = sorted(logs_dir.iterdir())
        click.secho(f"\nLightning 日志 ({len(versions)} 个版本):", fg="cyan")
        for v in versions:
            click.echo(f"  {v.name}")

    # Wandb 记录
    wandb_dirs = _find_wandb_runs(model_dir)
    if wandb_dirs:
        click.secho(f"\nW&B 记录 ({len(wandb_dirs)} 个):", fg="cyan")
        for wd in wandb_dirs:
            click.echo(f"  {wd}")


def _format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"
