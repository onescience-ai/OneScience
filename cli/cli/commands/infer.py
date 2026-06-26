import click
from ..core.runner import run_model, print_metrics
from ..core.registry import model_registry


@click.command("infer")
@click.argument("model_alias")
@click.option("-dataset", required=True, help="数据集名称或路径")
def infer(model_alias, dataset):
    """仅执行模型推理"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return
    click.secho(f"开始推理: {model_alias}", fg="green")
    click.secho(f"数据集: {dataset}", fg="green")
    r = run_model(model_alias, "infer", dataset)
    if r["success"]:
        click.secho(f"推理完成: {model_alias}", fg="green")
    else:
        err_msg = r.get("error") or r.get("output", "")
        click.secho(f"推理失败: {err_msg}", fg="red")
    print_metrics(r)
