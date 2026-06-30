import click
from pathlib import Path
from ..core.runner import run_model, collect_results, print_comparison, print_metrics
from ..core.registry import model_registry


# 各领域的默认数据集
_DOMAIN_DEFAULT_DATASETS = {
    "earth": "era5",
    "cfd": "airfoil",
    "biosciences": "evo2",
    "matchem": "mace",
}

# 模型 config.yaml 中 datapipe.name → CLI 数据集名映射（通用方案：直接转小写）
# 模型在 conf/config.yaml 中声明 datapipe.name，例如：
#   datapipe:
#     name: "ERA5"    → 使用数据集 "era5"
#     name: "CMEMS"   → 使用数据集 "cmems"


def _get_model_default_dataset(model_dir: Path) -> str | None:
    """从模型目录的 conf/config.yaml 中读取 datapipe.name，转小写作为数据集名"""
    config_path = model_dir / "conf" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        pipe_name = data.get("datapipe", {}).get("name", "")
        if not pipe_name:
            return None
        return pipe_name.lower()
    except Exception:
        return None


@click.command("bench")
@click.option("-dataset", default=None, help="数据集名称或路径")
@click.option("-models", default=None, help="模型别名列表，逗号分隔")
@click.option("--domain", default=None, help="按领域执行所有模型（如 earth/cfd/all）")
@click.option("--dir", "model_dir", default=None, help="按模型目录名执行该目录下所有模型（如 CFD_Benchmark）")
def bench(dataset, models, domain, model_dir):
    """使用指定数据集运行多个模型（训练+推理+评估）"""

    # ── 解析模型列表 ──────────────────────────────────
    aliases = []
    if models:
        aliases = [m.strip() for m in models.split(",") if m.strip()]
    elif domain:
        all_models = model_registry.list_models()
        if domain == "all":
            aliases = [m["alias"] for m in all_models]
        else:
            aliases = [m["alias"] for m in all_models if m.get("domain") == domain]
        if not aliases:
            click.secho(f"领域 '{domain}' 下没有找到可用模型", fg="red")
            return
    elif model_dir:
        all_models = model_registry.list_models()
        aliases = [m["alias"] for m in all_models if m.get("model") == model_dir]
        if not aliases:
            click.secho(f"目录 '{model_dir}' 下没有找到可用模型", fg="red")
            return
    else:
        click.secho("请指定 -models / --domain / --dir 参数", fg="red")
        return

    # ── 解析数据集 ──────────────────────────────────
    # 兼容旧用法：使用 -models 时 -dataset 必填
    if models and not dataset:
        click.secho("使用 -models 时必须指定 -dataset 参数", fg="red")
        return
    # --dir 必须指定 -dataset
    if model_dir and not dataset:
        click.secho("使用 --dir 时必须指定 -dataset 参数", fg="red")
        return

    # ── 执行模型 ──────────────────────────────────
    results = []

    if domain == "all" and not dataset:
        # -domain all 无 -dataset：按领域分组，每组用默认数据集
        models_by_domain: dict = {}
        for m in all_models:
            d = m.get("domain", "_custom")
            models_by_domain.setdefault(d, []).append(m["alias"])

        for d, domain_aliases in models_by_domain.items():
            default_ds = _DOMAIN_DEFAULT_DATASETS.get(d)
            if not default_ds:
                click.echo(f"  跳过领域 '{d}'（无默认数据集，请使用 -dataset 指定）")
                continue
            click.secho(f"\n{'=' * 56}", fg="cyan")
            click.secho(f"  领域: {d}  |  数据集: {default_ds}", fg="cyan")
            click.secho(f"{'=' * 56}", fg="cyan")
            for alias in domain_aliases:
                # 优先使用模型 config.yaml 中声明的数据集
                info = model_registry.resolve(alias)
                model_dataset = None
                if info and info.get("model_dir"):
                    model_dataset = _get_model_default_dataset(Path(info["model_dir"]))
                effective_ds = model_dataset or default_ds
                if model_dataset:
                    click.secho(f"  → {alias} 使用自声明数据集: {model_dataset}", fg="cyan")
                _run_single_model(alias, effective_ds, results)
    else:
        # 已有 dataset（由用户通过 -dataset 或在 -domain <特定> 时自动补全）
        effective_dataset = dataset
        if not effective_dataset and domain:
            effective_dataset = _DOMAIN_DEFAULT_DATASETS.get(domain)
        if not effective_dataset:
            click.secho(f"领域 '{domain}' 没有默认数据集，请通过 -dataset 指定", fg="red")
            return

        for alias in aliases:
            _run_single_model(alias, effective_dataset, results)

    collect_results(results)
    click.echo(f"所有模型执行完成")
    print_comparison(results)


def _run_single_model(alias: str, dataset: str, results: list):
    """执行单个模型并记录结果"""
    info = model_registry.resolve(alias)
    if not info:
        click.secho(f"未知模型: {alias}", fg="red")
        return
    click.echo(f"\n{'=' * 48}")
    click.secho(f"开始执行模型: {alias}", fg="green")
    click.secho(f"模型领域: {info['domain']}", fg="green")
    click.secho(f"数据集: {dataset}", fg="green")
    click.echo(f"{'=' * 48}")
    r = run_model(alias, "bench", dataset)
    results.append(r)
    if r["success"]:
        click.secho(f"模型执行完成: {alias}", fg="green")
    else:
        err_msg = r.get("error") or r.get("output", "")
        click.secho(f"模型执行失败: {err_msg}", fg="red")
    print_metrics(r)
    click.echo()
