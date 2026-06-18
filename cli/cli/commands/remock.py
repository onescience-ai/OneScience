import click
from ..core.runner import run_model
from ..core.registry import model_registry


@click.command("REMOCK")
@click.argument("model_alias")
def remock(model_alias):
    """重置模型环境"""
    if not model_registry.resolve(model_alias):
        click.secho(f"未知模型: {model_alias}", fg="red")
        return
    r = run_model(model_alias, "remock", "")
    if r["success"]:
        click.secho("环境重置完成", fg="green")
    else:
        click.secho(f"重置失败: {r.get('error', '')}", fg="red")
