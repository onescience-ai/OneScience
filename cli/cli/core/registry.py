"""
模型与模块注册中心。

提供两种模型发现方式：
  1. ModelRegistry — 旧版别名查找（向后兼容），resolve() 内部已集成 config 的 5 级优先级
  2. 全局单例 model_registry

模块注册 (ModuleRegistry) 保持不变。
"""

import os
import typing as t
from pathlib import Path

from .config import config, BUILTIN_MODELS_RAW, BUILTIN_DOMAINS, BUILTIN_DOMAIN_DIR_MAP


def _find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "setup.py").exists() and (parent / "examples").exists():
            return parent
        # 也通过 src/onescience 检测
        if (parent / "src" / "onescience").exists():
            return parent
    return cwd


PROJECT_ROOT = _find_project_root()
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src" / "onescience"
MODULES_DIR = SRC_DIR / "modules"


DOMAIN_DESCRIPTIONS = BUILTIN_DOMAINS
DOMAIN_DIR_MAP = BUILTIN_DOMAIN_DIR_MAP


def _is_model_dir(path: Path) -> bool:
    """检查目录是否是真正的模型目录（有训练/推理入口脚本）

    仅靠 conf/config.yaml 不足以判断（TJ-Data 等非模型目录也有配置文件），
    必须存在 Python 入口脚本才认为是模型。
    """
    for pattern in ("train*.py", "finetune*.py", "cli*.py", "run*.py"):
        if list(path.glob(pattern)):
            return True
    return False


MODULE_TYPE_DESC = {
    "attention": "注意力机制",
    "embedding": "嵌入层",
    "transformer": "变换器",
    "pooling": "池化层",
    "linear": "线性层",
    "fuser": "融合层",
    "recovery": "恢复层",
    "mlp": "多层感知机",
    "fourier": "傅里叶变换",
    "encoder": "编码器",
    "decoder": "解码器",
    "head": "输出头",
    "edge": "边特征",
    "node": "节点特征",
    "processor": "处理器",
    "sample": "采样",
    "equivariant": "等变变换",
    "fc": "全连接",
    "afno": "AFNO",
    "diffusion": "扩散模型",
    "msa": "多序列比对",
    "pairformer": "Pairformer",
}


def get_model_dir(info: dict) -> Path:
    """从模型解析结果中获取模型目录路径

    Args:
        info: model_registry.resolve() 的返回结果

    Returns:
        模型目录的 Path 对象
    """
    model_dir = info.get("model_dir")
    if model_dir:
        return Path(model_dir)
    return EXAMPLES_DIR / info["domain"] / info["model"]


class ModelRegistry:
    """模型注册中心

    向后兼容: 保留原有的 _aliases 和 resolve() 方法，
    resolve() 内部优先使用 config 的 5 级模型发现，
    未命中时回退到内置别名。
    """

    def __init__(self):
        self._aliases: dict[str, tuple[str, str, str]] = {}
        self._descriptions: dict[str, str] = {}
        self._load_builtin()

    def _load_builtin(self):
        """从内置预设加载模型"""
        for alias, (domain, model_dir, sub_model, desc) in BUILTIN_MODELS_RAW.items():
            self._aliases[alias] = (domain, model_dir, sub_model)
            self._descriptions[alias] = desc

    def resolve(self, name: str) -> t.Optional[dict]:
        """5 级优先级查找模型

        1. 完整路径 (含 / 或 \\)
        2. 自定义模型 (config.models)
        3. 扫描目录 (config.model_roots)
        4. 当前工作目录
        5. 内置预设别名
        6. 目录名自动发现 (向后兼容)
        """
        name_lower = name.lower().strip()

        # 级别 1-4: config 驱动
        result = config.resolve_model(name)
        if result:
            # 将 config 返回格式转为 ModelRegistry 格式
            return {
                "alias": result["alias"],
                "domain": result["domain"],
                "model": result["model"],
                "sub_model": result.get("sub_model", ""),
                "description": result.get("description", ""),
                "model_dir": result.get("model_dir"),
                "source": result.get("source", "builtin"),
            }

        # 级别 5: 内置别名
        if name_lower in self._aliases:
            domain, model_dir, sub_model = self._aliases[name_lower]
            return {
                "alias": name_lower,
                "domain": domain,
                "model": model_dir,
                "sub_model": sub_model,
                "description": self._descriptions.get(name_lower, ""),
                "model_dir": EXAMPLES_DIR / DOMAIN_DIR_MAP.get(domain, domain) / model_dir if EXAMPLES_DIR.exists() else None,
                "source": "builtin",
            }

        # 级别 6: 目录名自动发现 (旧版兼容)
        found = self._discover_by_name(name_lower)
        if found:
            return found

        return None

    def _discover_by_name(self, name: str) -> t.Optional[dict]:
        """在 examples 目录下按名称发现模型（旧版兼容）"""
        if not EXAMPLES_DIR.exists():
            return None
        for domain_dir in EXAMPLES_DIR.iterdir():
            if not domain_dir.is_dir() or domain_dir.name == "configs":
                continue
            domain_key = DOMAIN_DIR_MAP.get(domain_dir.name)
            if not domain_key:
                continue
            for model_dir in domain_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name == "conf" or not _is_model_dir(model_dir):
                    continue
                model_lower = model_dir.name.lower()
                no_underscore = model_lower.replace("_", "")
                if model_lower == name or no_underscore == name:
                    return {
                        "alias": name,
                        "domain": domain_key,
                        "model": model_dir.name,
                        "sub_model": "",
                        "description": "自动发现的模型",
                        "model_dir": model_dir,
                        "source": "discover",
                    }
        return None

    def list_models(self, domain: t.Optional[str] = None) -> t.List[dict]:
        """列出所有可用模型

        合并内置 + 配置文件自定义 + 扫描目录发现 + 旧版 examples 自动发现
        """
        results = config.list_models(domain)

        # 旧版兼容: 扫描 examples 目录下未注册的模型
        if EXAMPLES_DIR.exists():
            seen_keys = {f"{r['domain']}|{r['model']}|{r.get('sub_model', '')}" for r in results}
            for domain_dir in EXAMPLES_DIR.iterdir():
                if not domain_dir.is_dir() or domain_dir.name == "configs":
                    continue
                domain_key = DOMAIN_DIR_MAP.get(domain_dir.name)
                if not domain_key or (domain and domain_key != domain):
                    continue
                for model_dir in domain_dir.iterdir():
                    if not model_dir.is_dir() or model_dir.name == "conf" or not _is_model_dir(model_dir):
                        continue
                    key = f"{domain_key}|{model_dir.name}|"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append({
                            "alias": model_dir.name.lower().replace("_", ""),
                            "model": model_dir.name,
                            "domain": domain_key,
                            "domain_desc": DOMAIN_DESCRIPTIONS.get(domain_key, ""),
                            "description": "自动发现的模型",
                            "sub_model": "",
                            "source": "discover",
                        })

        return results


class ModuleRegistry:
    """模块注册中心（不变）"""

    def __init__(self):
        self._modules: t.List[dict] = []
        self._load()

    def _load(self):
        raw = [
            ("OneAttention", "attention", "统一注意力机制接口，支持多种注意力实现"),
            ("SelfAttention", "attention", "标准自注意力机制"),
            ("MultiHeadAttention", "attention", "多头注意力机制"),
            ("LinearAttention", "attention", "线性注意力，降低计算复杂度"),
            ("FlashAttention", "attention", "Flash Attention，高效注意力计算"),
            ("FactAttention", "attention", "因子分解注意力"),
            ("EarthAttention2D", "attention", "2D地球科学专用注意力"),
            ("EarthAttention3D", "attention", "3D地球科学专用注意力"),
            ("OneEmbedding", "embedding", "统一嵌入层接口"),
            ("PanguEmbedding", "embedding", "盘古模型嵌入层"),
            ("FuxiEmbedding", "embedding", "伏羲模型嵌入层"),
            ("FourCastNetEmbedding", "embedding", "FourCastNet嵌入层"),
            ("FengwuEncoder", "embedding", "风乌模型编码器"),
            ("XiheEmbedding", "embedding", "羲和模型嵌入层"),
            ("OneTransformer", "transformer", "统一变换器接口"),
            ("FuxiTransformer", "transformer", "伏羲模型变换器"),
            ("XiheTransformer", "transformer", "羲和模型变换器"),
            ("XiheLocalTransformer", "transformer", "羲和局部变换器"),
            ("EarthTransformer2DBlock", "transformer", "2D地球科学变换器块"),
            ("EarthTransformer3DBlock", "transformer", "3D地球科学变换器块"),
            ("PreLNTransformerBlock", "transformer", "Pre-LayerNorm变换器块"),
            ("SwinTransformerBlock", "transformer", "Swin变换器块"),
            ("GNOTTransformerBlock", "transformer", "GNOT变换器块"),
            ("GalerkinTransformerBlock", "transformer", "伽辽金变换器块"),
            ("TransolverBlock", "transformer", "Transolver块"),
            ("NeuralSpectralBlock", "transformer", "神经谱块"),
            ("ProtenixTransformer", "transformer", "Protenix变换器"),
            ("FactformerBlock", "transformer", "Factformer块"),
            ("OrthogonalNeuralBlock", "transformer", "正交神经块"),
            ("OnePooling", "pooling", "统一池化接口"),
            ("RNNClusterPooling", "pooling", "RNN聚类池化"),
            ("OneLinear", "linear", "统一线性层接口"),
            ("OneFuser", "fuser", "统一融合层接口"),
            ("PanguFuser", "fuser", "盘古模型融合层"),
            ("FuxiFuser", "fuser", "伏羲模型融合层"),
            ("FourCastNetFuser", "fuser", "FourCastNet融合层"),
            ("FengwuFuser", "fuser", "风乌模型融合层"),
            ("XiheFuser", "fuser", "羲和模型融合层"),
            ("XiheLocalSIEFuser", "fuser", "羲和局部SIE融合层"),
            ("XiheGlobalSIEFuser", "fuser", "羲和全局SIE融合层"),
            ("OneRecovery", "recovery", "统一恢复层接口"),
            ("PanguPatchRecovery", "recovery", "盘古模型补丁恢复"),
            ("XihePatchRecovery", "recovery", "羲和模型补丁恢复"),
            ("PanguUpsample", "recovery", "盘古模型上采样"),
            ("FuxiUpsample", "recovery", "伏羲模型上采样"),
            ("XiheUpsample", "recovery", "羲和模型上采样"),
            ("PanguDownsample", "recovery", "盘古模型下采样"),
            ("FuxiDownsample", "recovery", "伏羲模型下采样"),
            ("OneMlp", "mlp", "统一MLP接口"),
            ("XiheMLP", "mlp", "羲和模型MLP"),
            ("OneFourier", "fourier", "统一傅里叶变换接口"),
            ("FourCastNetAFNO", "fourier", "FourCastNet AFNO"),
            ("GroupSpectral", "fourier", "群谱变换"),
            ("GroupConv", "fourier", "群卷积"),
            ("OneEncoder", "encoder", "统一编码器接口"),
            ("OneDecoder", "decoder", "统一解码器接口"),
            ("FengwuDecoder", "decoder", "风乌模型解码器"),
            ("OneHead", "head", "统一输出头接口"),
            ("OneEdge", "edge", "统一边特征接口"),
            ("OneNode", "node", "统一节点特征接口"),
            ("MeshNodeBlock", "node", "网格节点块"),
            ("OneProcessor", "processor", "统一处理器接口"),
            ("BiStrideProcessor", "processor", "双步长处理器"),
            ("OneSample", "sample", "统一采样接口"),
            ("OneEquivariant", "equivariant", "等变变换接口"),
            ("OneFC", "fc", "全连接层接口"),
            ("OneAFNO", "afno", "AFNO接口"),
            ("OneDiffusion", "diffusion", "扩散模型接口"),
            ("OneMSA", "msa", "多序列比对接口"),
            ("OnePairformer", "pairformer", "Pairformer接口"),
        ]
        for name, typ, desc in raw:
            self._modules.append({"name": name, "type": typ, "description": desc})

    def list(self, type_filter: t.Optional[str] = None) -> t.List[dict]:
        if type_filter:
            return [m for m in self._modules if m["type"] == type_filter]
        return self._modules


# 全局单例
model_registry = ModelRegistry()
module_registry = ModuleRegistry()
