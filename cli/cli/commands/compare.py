import click
from ..core.runner import load_saved_results, print_comparison


@click.command("compare")
@click.argument("model_aliases", required=False)
@click.option("-format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
def compare(model_aliases, fmt):
    """对比多个模型的执行结果

    从之前执行保存的结果中加载指标数据并对比展示。
    如果不指定模型别名，则对比所有已保存结果的模型。
    """
    if model_aliases:
        aliases = [a.strip() for a in model_aliases.split(",") if a.strip()]
    else:
        aliases = None

    results = load_saved_results(aliases)

    if not results:
        click.echo("没有找到可对比的模型结果。请先执行 bench 或 train 命令。")
        return

    click.secho("找到 {} 个模型的结果".format(len(results)), fg="green")
    for r in results:
        status = "OK" if r.get("metrics") else "--"
        click.echo("  {}  {}  [{}] {}".format(
            status, r["alias"].ljust(20), r["domain"], r["model"]
        ))

    if fmt == "json":
        import json
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
    elif fmt == "csv":
        for r in results:
            m = r["metrics"]
            line = r["alias"]
            for k, v in m.items():
                line += "," + str(v)
            click.echo(line)
    else:
        print_comparison(results)
