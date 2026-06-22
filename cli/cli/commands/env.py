import os
import json
import click
from pathlib import Path
from ..core.config import config, _find_config_file, BUILTIN_NAME


def _get_config_path() -> Path:
    """获取 .onescience.json 路径（优先找已有，否则用 cwd）"""
    existing = _find_config_file()
    if existing:
        return existing
    return Path.cwd() / ".onescience.json"


@click.group("env")
def env_group():
    """环境管理（项目切换与配置）"""


@env_group.command("list")
def list_envs():
    """列出可用环境和当前环境"""
    click.secho("环境列表", fg="green")

    current_name = config.name
    config_file = config.file_path

    # 当前环境
    click.secho(f"\n当前环境: {current_name}", fg="cyan", bold=True)
    if config_file:
        click.echo(f"  配置文件: {config_file}")
    else:
        click.echo("  使用内置预设 (无需配置文件)")

    # 内置预设
    click.echo(f"\n内置预设:")
    click.echo(f"  {BUILTIN_NAME:20} - OneScience 官方模型 (30+ 模型)")

    # 如果已有配置文件，显示其信息
    if config_file:
        click.echo(f"\n自定义配置:")
        click.echo(f"  项目名称: {config.name}")
        custom_models = config.custom_models
        if custom_models:
            click.echo(f"  自定义模型: {len(custom_models)} 个")
            for alias in sorted(custom_models.keys()):
                click.echo(f"    - {alias}")
        model_roots = config.model_roots
        if model_roots:
            click.echo(f"  扫描目录: {len(model_roots)} 个")
            for r in model_roots:
                click.echo(f"    - {r}")

    click.echo("\n切换环境:")
    click.echo("  onescience env use <name>")
    click.echo("初始化新项目:")
    click.echo("  onescience env init")


@env_group.command("use")
@click.argument("name", required=False, default="onescience")
def use_env(name):
    """切换到指定环境

    onescience      → 使用内置预设（无需配置文件）
    <项目名称>       → 使用自定义配置（需先执行 env init）
    """
    if name == BUILTIN_NAME:
        # 切回内置预设：删除或停用 .onescience.json
        config_file = _get_config_path()
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                data.pop("name", None)
                config_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                click.secho(f"已切换到内置预设 (onescience)", fg="green")
            except Exception as e:
                click.secho(f"切换失败: {e}", fg="red")
                return
            click.echo(f"配置文件保留在: {config_file}")
        else:
            click.secho(f"已切换到内置预设 (onescience)", fg="green")
        return

    # 查找配置文件
    config_file = _find_config_file()
    if not config_file:
        click.secho(f"未找到 .onescience.json 配置文件", fg="red")
        click.echo("请先执行: onescience env init")
        return

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        data["name"] = name
        config_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.secho(f"已切换到环境: {name}", fg="green")
        click.echo(f"配置文件: {config_file}")
    except Exception as e:
        click.secho(f"切换环境失败: {e}", fg="red")


@env_group.command("init")
@click.option("-name", default="my_project", help="项目名称")
@click.option("-roots", default="", help="模型扫描目录，逗号分隔")
@click.option("-models-dir", "models_dir", default="", help="自定义数据集目录")
def init_env(name, roots, models_dir):
    """初始化当前目录为新的项目环境"""
    config_path = _get_config_path()

    if config_path.exists():
        if not click.confirm(f"配置文件已存在: {config_path}\n是否覆盖?"):
            return

    data = {
        "name": name,
        "model_roots": [],
        "models": {},
        "datasets": {},
        "domains": {},
    }

    if roots:
        data["model_roots"] = [r.strip() for r in roots.split(",") if r.strip()]

    if models_dir:
        data["datasets_dir"] = models_dir

    try:
        config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.secho(f"项目初始化完成", fg="green")
        click.echo(f"  配置文件: {config_path}")
        click.echo(f"  项目名称: {name}")
        click.echo(f"  模型目录: {data['model_roots'] or '（未设置）'}")
        click.echo(f"\n下一步:")
        click.echo(f"  1. 编辑 {config_path.name} 添加自定义模型和数据集")
        click.echo(f"  2. onescience env use {name} 切换到当前环境")
        click.echo(f"  3. 添加模型: 在 models 下注册或在 model_roots 目录下放模型文件夹")
    except Exception as e:
        click.secho(f"初始化失败: {e}", fg="red")


@env_group.command("info")
def env_info():
    """显示当前环境的详细信息"""
    click.secho(f"当前环境: {config.name}", fg="green", bold=True)

    if config.file_path:
        click.echo(f"配置文件: {config.file_path}")
    else:
        click.echo("配置来源: 内置预设 (onescience)")

    click.echo("")
    custom_models = config.custom_models
    if custom_models:
        click.secho("自定义模型:", fg="cyan")
        for alias, meta in sorted(custom_models.items()):
            domain = meta.get("domain", "_")
            desc = meta.get("description", alias)
            click.echo(f"  {alias:<20} [{domain}] {desc}")

    model_roots = config.model_roots
    if model_roots:
        click.secho("\n扫描目录:", fg="cyan")
        for r in model_roots:
            status = "✓" if r.exists() else "✗"
            click.echo(f"  {status} {r}")

    custom_ds = config._data.get("datasets", {})
    if custom_ds:
        click.secho("\n自定义数据集:", fg="cyan")
        for name, path in sorted(custom_ds.items()):
            exists = "✓" if os.path.exists(path) else "✗"
            click.echo(f"  {name:<20} {exists} {path}")
