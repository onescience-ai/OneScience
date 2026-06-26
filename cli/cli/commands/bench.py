import click
from ..core.runner import run_model, collect_results, print_comparison, print_metrics
from ..core.registry import model_registry


@click.command("bench")
@click.option("-dataset", required=True, help="数据集名称或路径")
@click.option("-models", required=True, help="模型别名列表，逗号分隔")
def bench(dataset, models):
    """使用指定数据集运行多个模型（训练+推理+评估）"""
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    results = []
    for alias in aliases:
        info = model_registry.resolve(alias)
        if not info:
            click.secho(f"未知模型: {alias}", fg="red")
            continue
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
    collect_results(results)
    click.echo(f"所有模型执行完成，数据集: {dataset}")
    print_comparison(results)
