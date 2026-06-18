import os
import click
from pathlib import Path
from ..core.registry import model_registry, module_registry, DOMAIN_DESCRIPTIONS
from ..core.formatter import Formatter


@click.group("list")
def list_group():
    """列出可用资源"""


@list_group.command("models")
@click.option("-domain", default=None, help="按领域筛选")
@click.option("-model", "model_filter", default=None, help="按模型别名筛选")
@click.option("-all", "show_all", is_flag=True, help="显示详细信息")
@click.option("-format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
def list_models(domain, model_filter, show_all, fmt):
    """列出所有可用模型"""
    models = model_registry.list_models(domain)
    if model_filter:
        models = [m for m in models if m["alias"] == model_filter.lower()]
    if not models:
        click.echo("没有找到匹配的模型")
        return
    if fmt == "json":
        click.echo(Formatter.json(models))
        return
    headers = ["模型别名", "模型名称", "领域", "描述"] if show_all else ["模型别名", "模型名称"]
    rows = []
    for m in models:
        if show_all:
            rows.append([m["alias"], m["model"], m["domain_desc"], m["description"]])
        else:
            rows.append([m["alias"], m["model"]])
    click.echo(Formatter.format_output(models, headers, rows, fmt))


@list_group.command("modules")
@click.option("-type", "type_filter", default=None, help="按类型筛选")
@click.option("-all", "show_all", is_flag=True, help="显示详细信息")
@click.option("-format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
def list_modules(type_filter, show_all, fmt):
    """列出所有可用模块"""
    modules = module_registry.list(type_filter)
    if not modules:
        click.echo("没有找到匹配的模块")
        return
    if fmt == "json":
        click.echo(Formatter.json(modules))
        return
    headers = ["模块名称", "类型", "描述"] if show_all else ["模块名称", "类型"]
    rows = []
    for m in modules:
        if show_all:
            rows.append([m["name"], m["type"], m["description"]])
        else:
            rows.append([m["name"], m["type"]])
    click.echo(Formatter.format_output(modules, headers, rows, fmt))


@list_group.command("datasets")
@click.option("-format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
@click.option("-full", is_flag=True, help="扫描实际数据集目录")
def list_datasets(fmt, full):
    """列出可用数据集"""
    if full:
        datasets_dir = os.environ.get("ONESCIENCE_DATASETS_DIR", "")
        if not datasets_dir or not Path(datasets_dir).exists():
            click.echo("未设置 ONESCIENCE_DATASETS_DIR 或目录不存在")
            return
        click.echo(f"数据集目录: {datasets_dir}")
        for d in Path(datasets_dir).iterdir():
            if d.is_dir():
                try:
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    items = len(list(d.iterdir()))
                    for unit in ["B", "KB", "MB", "GB"]:
                        if size < 1024:
                            click.echo(f"  {d.name:<25} {size:.1f}{unit:>7}  {items} 个文件")
                            break
                        size /= 1024
                except (OSError, PermissionError):
                    click.echo(f"  {d.name:<25}  (无权限)")
        return
    from ..road import DATASET_PATHS, ONESCIENCE_DATASETS_DIR
    rows = []
    for name, path in sorted(DATASET_PATHS.items()):
        full_path = str(Path(ONESCIENCE_DATASETS_DIR) / path) if ONESCIENCE_DATASETS_DIR else ""
        exists = os.path.isdir(full_path) if full_path else False
        rows.append([name, path, "✓" if exists else ""])
    headers = ["名称", "相对路径", "本地存在"]
    click.echo(Formatter.format_output(rows, headers, rows, fmt))
