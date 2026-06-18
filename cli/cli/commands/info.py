import click
from ..core.registry import MODULES_DIR, MODULE_TYPE_DESC


_FALLBACK_DESCRIPTIONS = {
    "OneEmbedding": "嵌入层模块，用于将输入特征映射到高维空间",
    "OneAttention": "注意力机制模块，实现多头自注意力机制",
    "OneTransformer": "Transformer模块，实现标准Transformer架构",
    "OnePooling": "池化层模块，支持多种池化策略",
    "OneLinear": "线性层模块，实现线性变换",
    "OneFuser": "特征融合模块，融合多模态特征",
    "OneRecovery": "特征恢复模块，恢复被压缩的特征",
    "OneMlp": "多层感知器模块，实现深度神经网络",
    "OneFourier": "傅里叶变换模块，实现频域变换",
    "OneEncoder": "编码器模块，实现序列编码",
    "OneDecoder": "解码器模块，实现序列解码",
    "OneHead": "输出头模块，生成最终输出",
    "OneEdge": "边处理模块，处理图结构中的边信息",
    "OneNode": "节点处理模块，处理图结构中的节点信息",
    "OneProcessor": "处理器模块，执行核心计算逻辑",
    "OneSample": "采样模块，实现数据采样策略",
    "OneEquivariant": "等变模块，保持对称性变换",
    "OneFC": "全连接模块，实现全连接层",
    "OneAFNO": "自适应傅里叶神经算子，处理网格数据",
    "OneDiffusion": "扩散模型模块，实现扩散生成模型",
    "OneMSA": "多头注意力模块，实现多头注意力机制",
    "OnePairformer": "配对Transformer模块，处理配对数据",
}


@click.group("info")
def info_group():
    """显示资源详细信息"""


@info_group.command("module")
@click.argument("module_names", required=True)
def show_module(module_names):
    """显示模块信息或代码"""
    names = [n.strip() for n in module_names.split(",") if n.strip()]
    for i, name in enumerate(names):
        if i > 0:
            click.echo()
        _show_single_module(name)


def _show_single_module(name: str):
    if not MODULES_DIR.exists():
        click.echo(f"模块目录不存在: {MODULES_DIR}")
        return
    found_path = None
    module_type = ""
    for type_dir in MODULES_DIR.iterdir():
        if not type_dir.is_dir():
            continue
        for f in type_dir.iterdir():
            if f.suffix == ".py" and f.stem.lower().startswith(name.lower()):
                found_path = f
                module_type = type_dir.name
                break
        if found_path:
            break
    if not found_path:
        for f in MODULES_DIR.rglob("*.py"):
            if f.stem.lower() == name.lower():
                found_path = f
                module_type = f.parent.name
                break
    if found_path:
        click.secho(f"模块名称: {name}", fg="green")
        click.secho(f"模块类型: {MODULE_TYPE_DESC.get(module_type, module_type)}", fg="green")
        click.secho(f"文件路径: {found_path}", fg="green")
        click.echo()
        click.secho("【模块代码】", fg="cyan")
        click.echo(found_path.read_text(encoding="utf-8"))
    else:
        camel_name = f"One{name[0].upper()}{name[1:]}"
        desc = _FALLBACK_DESCRIPTIONS.get(camel_name, "暂无详细描述")
        click.secho(f"模块名称: {name}", fg="green")
        click.secho(f"类名: {camel_name}", fg="green")
        click.secho(f"描述: {desc}", fg="green")
        click.echo()
        click.echo(f"模块可能位于: {MODULES_DIR}/")
