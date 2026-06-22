import click
from ..core.runner import run_model, print_metrics
from ..core.registry import model_registry


@click.command("eval")
@click.argument("model_alias")
@click.option("-dataset", required=True, help="数据集名称或路径")
@click.option("-metrics-type", "metrics_type", default=None,
              type=click.Choice(["earth", "cfd", "bio", "matchem", "auto"]),
              help="指标类型（默认自动检测）")
def eval(model_alias, dataset, metrics_type):
    """仅执行模型评估"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return
    click.secho(f"开始评估: {model_alias}", fg="green")
    click.secho(f"数据集: {dataset}", fg="green")
    if metrics_type and metrics_type != "auto":
        click.secho(f"指标类型: {metrics_type}", fg="green")
    r = run_model(model_alias, "eval", dataset)
    if r["success"]:
        click.secho(f"评估完成: {model_alias}", fg="green")
    else:
        err_msg = r.get("error") or r.get("output", "")
        if len(err_msg) > 200:
            err_msg = err_msg[:200] + "..."
        click.secho(f"评估失败: {err_msg}", fg="red")
    print_metrics(r)
