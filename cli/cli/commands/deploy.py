import click
from ..core.registry import model_registry, get_model_dir


@click.group("deploy")
def deploy_group():
    """模型部署（ONNX 导出与推理服务）

    TODO: 当前为使用指南模式，后续需实现自动化导出和服务启动。
    """


@deploy_group.command("export")
@click.argument("model_alias")
@click.option("-format", "export_format", default="onnx", type=click.Choice(["onnx"]),
              help="导出格式（目前支持 ONNX）")
def export_model(model_alias, export_format):
    """导出模型为部署格式

    将训练好的 PyTorch 模型导出为 ONNX 格式，用于生产环境部署。
    """
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return

    model_dir = get_model_dir(info)
    if not model_dir.exists():
        click.secho(f"模型目录不存在: {model_dir}", fg="red")
        return

    click.secho(f"模型导出: {model_alias}", fg="green")
    click.echo(f"  模型: {info['model']} ({info['domain']})")
    click.echo(f"  目录: {model_dir}")
    click.echo(f"  格式: {export_format.upper()}")
    click.echo("")
    click.echo("导出方式（在 Python 中调用）:")
    click.echo("  from onescience.deploy.onnx import export_to_onnx_stream")
    click.echo("  model = torch.load('{}/model.pth')".format(model_dir))
    click.echo("  onnx_bytes = export_to_onnx_stream(model, dummy_input)")
    click.echo("  with open('model.onnx', 'wb') as f:")
    click.echo("      f.write(onnx_bytes)")
    click.echo("")
    click.echo("或者参考:")
    click.echo("  from onescience.deploy.onnx.utils import export_to_onnx_stream")


@deploy_group.command("serve")
@click.argument("model_alias")
@click.option("-port", default=8000, type=int, help="服务端口")
@click.option("-backend", default="triton", type=click.Choice(["triton"]),
              help="推理后端")
def serve_model(model_alias, port, backend):
    """启动模型推理服务（Triton）"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return

    model_dir = get_model_dir(info)
    click.secho(f"推理服务: {model_alias}", fg="green")
    click.echo(f"  模型: {info['model']} ({info['domain']})")
    click.echo(f"  目录: {model_dir}")
    click.echo(f"  后端: {backend}")
    click.echo(f"  端口: {port}")
    click.echo("")
    click.echo("Triton 推理服务启动方式:")
    click.echo("  tritonserver --model-repository=<model_repo> \\")
    click.echo("              --http-port={} \\".format(port))
    click.echo("              --grpc-port={}".format(port + 1))
    click.echo("")
    click.echo("Python 客户端调用:")
    click.echo("  import tritonclient.http as httpclient")
    click.echo("  client = httpclient.InferenceServerClient(url='localhost:{}')".format(port))
    click.echo("  # 参考 onescience.deploy.triton 模块")
