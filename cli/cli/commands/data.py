import os
import click
from pathlib import Path
from ..road import DATASET_PATHS, ONESCIENCE_DATASETS_DIR, list_available_datasets


DOMAIN_DATASETS = {
    "earth": ["era5", "era5_stats", "era5_static", "cwb", "graphcast"],
    "cfd": ["airfoil", "darcy", "elasticity", "ns", "pipe", "plasticity",
            "deepcfd", "beno", "pdenneval", "lagrangian_mgn", "topology"],
    "biosciences": ["evo2", "protenix", "openfold"],
    "matchem": ["mace", "matris"],
}


@click.group("data")
def data_group():
    """数据准备与管理"""


@data_group.command("list")
@click.option("-domain", default=None, help="按领域筛选 (earth/cfd/biosciences/matchem)")
@click.option("-format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
def list_datasets(domain, fmt):
    """列出所有可用数据集"""
    from ..core.formatter import Formatter

    available = list_available_datasets()
    rows = []
    for name, path in sorted(available.items()):
        if domain:
            domain_list = [d for d, names in DOMAIN_DATASETS.items() if name in names]
            if domain not in domain_list:
                continue
        alias_in = name in DATASET_PATHS
        rows.append([name, path, "注册" if alias_in else "未注册"])

    if not rows:
        click.echo("未找到数据集（ONESCIENCE_DATASETS_DIR 目录不可用）")
        click.echo(f"使用 road.py 中注册的数据集作为参考:")
        for d, names in DOMAIN_DATASETS.items():
            click.echo(f"  [{d}] {', '.join(names)}")
        return

    headers = ["名称", "路径", "状态"]
    if fmt == "json":
        click.echo(Formatter.json(rows))
    elif fmt == "csv":
        click.echo(Formatter.csv(headers, rows))
    else:
        click.echo(Formatter.table(headers, rows))


@data_group.command("info")
@click.argument("name")
def dataset_info(name):
    """显示数据集详情"""
    from ..road import get_dataset_path

    if name in DATASET_PATHS:
        rel_path = DATASET_PATHS[name]
        click.secho(f"数据集: {name}", fg="green")
        click.echo(f"  相对路径: {rel_path}")
        domain = None
        for d, names in DOMAIN_DATASETS.items():
            if name in names:
                domain = d
                break
        if domain:
            click.echo(f"  所属领域: {domain}")
        try:
            full_path = get_dataset_path(name)
            click.echo(f"  完整路径: {full_path}")
            click.echo(f"  本地存在: {'✓' if os.path.isdir(full_path) else '✗'}")
            if os.path.isdir(full_path):
                items = len(os.listdir(full_path))
                click.echo(f"  子项数量: {items}")
        except FileNotFoundError:
            click.echo(f"  完整路径: {str(Path(ONESCIENCE_DATASETS_DIR) / rel_path)}")
            click.echo(f"  本地存在: ✗（数据集目录未挂载）")
    else:
        try:
            full_path = get_dataset_path(name)
            click.secho(f"数据集: {name}", fg="green")
            click.echo(f"  完整路径: {full_path}")
            click.echo(f"  类型: 目录（未注册的数据集）")
            if os.path.isdir(full_path):
                items = len(os.listdir(full_path))
                click.echo(f"  子项数量: {items}")
        except FileNotFoundError:
            click.secho(f"数据集 '{name}' 未找到", fg="red")


@data_group.command("download")
@click.argument("name")
@click.option("-output", default=None, help="输出目录（默认使用数据集目录）")
@click.option("-year", default=None, help="指定年份（ERA5 数据集）")
def download_data(name, output, year):
    """下载数据集（目前支持 ERA5）"""
    if name == "era5":
        script_dir = None
        from ..core.registry import PROJECT_ROOT
        candidate = PROJECT_ROOT / "examples" / "earth" / "era5_dataset_prepare"
        if candidate.exists():
            script_dir = candidate

        if not script_dir:
            click.echo("未找到 ERA5 下载脚本（expected at examples/earth/era5_dataset_prepare/）")
            return

        output_dir = Path(output) if output else script_dir / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        click.secho(f"ERA5 数据下载与准备", fg="green")
        click.echo(f"  脚本目录: {script_dir}")
        click.echo(f"  输出目录: {output_dir}")
        click.echo(f"\n该过程包含 4 个步骤:")
        click.echo(f"  1. step_1_data_download.py  - 下载原始数据")
        click.echo(f"  2. step_2_data_conversion.py - 格式转换")
        click.echo(f"  3. step_3_data_merge.py     - 数据合并")
        click.echo(f"  4. step_4_stats_calculate.py - 统计计算")
        click.echo(f"\n请按顺序执行:")
        click.echo(f"  cd {script_dir}")
        click.echo(f"  python step_1_data_download.py [--year {year or 'YYYY'}]")
        click.echo(f"  python step_2_data_conversion.py")
        click.echo(f"  python step_3_data_merge.py")
        click.echo(f"  python step_4_stats_calculate.py")
    else:
        click.echo(f"数据集 '{name}' 暂不支持自动下载")
        click.echo(f"当前支持的下载: era5")


@data_group.command("generate")
@click.argument("model_alias")
@click.option("-output", default=None, help="输出目录（默认使用模型目录下的 data/）")
def generate_fake(model_alias, output):
    """生成模型的假数据（用于测试运行流程）"""
    from ..core.registry import model_registry, EXAMPLES_DIR

    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return

    model_dir = info.get("model_dir")
    if not model_dir:
        model_dir = EXAMPLES_DIR / info["domain"] / info["model"]
    model_dir = Path(model_dir)
    fake_script = model_dir / "fake_data.py"

    if not fake_script.exists():
        click.echo(f"模型 '{model_alias}' 没有 fake_data.py，无法生成假数据")
        click.echo(f"  查找位置: {fake_script}")
        return

    click.secho(f"生成 {model_alias} 假数据", fg="green")
    click.echo(f"  模型目录: {model_dir}")
    click.echo(f"  生成脚本: {fake_script}")

    cmd = ["python", "fake_data.py"]
    if output:
        cmd.extend(["--output", output])

    click.echo(f"  执行: {' '.join(cmd)}")
    click.echo(f"\n请在模型目录下手动执行:")
    click.echo(f"  cd {model_dir}")
    click.echo(f"  python fake_data.py" + (f" --output {output}" if output else ""))
