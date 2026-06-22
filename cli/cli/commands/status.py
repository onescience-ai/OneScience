import click
from pathlib import Path
from ..core.registry import model_registry, EXAMPLES_DIR
from ..core.runner import RESULTS_DIR


@click.command("status")
@click.argument("model_aliases", required=False)
def status(model_aliases):
    """查看模型执行状态"""
    if model_aliases:
        aliases = [a.strip() for a in model_aliases.split(",") if a.strip()]
    else:
        models = model_registry.list_models()
        aliases = [m["alias"] for m in models]
    click.secho("模型执行状态", fg="green")
    for alias in aliases:
        info = model_registry.resolve(alias)
        if not info:
            continue
        model_dir = info.get("model_dir")
        if not model_dir:
            model_dir = EXAMPLES_DIR / info["domain"] / info["model"]
        model_dir = Path(model_dir)
        result_dir = RESULTS_DIR / alias
        log_count = len(list(model_dir.glob("*.log"))) + len(list(model_dir.glob("*.log.*")))
        result_count = len(list(result_dir.rglob("*"))) if result_dir.exists() else 0
        if log_count > 0 and result_count > 0:
            status_str = "已完成"
        elif log_count > 0:
            status_str = "已运行"
        else:
            status_str = "未运行"
        click.echo(f"  {alias:<20} {status_str:<8} {log_count}个日志  {result_count}个结果")
