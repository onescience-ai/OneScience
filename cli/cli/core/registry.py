import os
import typing as t
from pathlib import Path


def _find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "setup.py").exists() and (parent / "examples").exists():
            return parent
    return cwd


PROJECT_ROOT = _find_project_root()
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src" / "onescience"
MODULES_DIR = SRC_DIR / "modules"


DOMAIN_DESCRIPTIONS = {
    "earth": "气象与气候",
    "cfd": "计算流体力学",
    "biosciences": "生物科学",
    "matchem": "材料化学",
    "structural": "结构力学",
}

DOMAIN_DIR_MAP = {
    "earth": "earth",
    "cfd": "cfd",
    "biosciences": "biosciences",
    "bio": "biosciences",
    "matchem": "matchem",
    "MaterialsChemistry": "matchem",
    "structural": "structural",
}


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


class ModelRegistry:
    def __init__(self):
        self._aliases: dict[str, tuple[str, str, str]] = {}
        self._descriptions: dict[str, str] = {}
        self._load_builtin()

    def _load_builtin(self):
        raw = {
            "pangu": ("earth", "pangu_weather", "盘古气象大模型，用于天气预报和气候预测"),
            "fourcastnet": ("earth", "fourcastnet", "FourCastNet，基于傅里叶神经算子的全球天气预报模型"),
            "fuxi": ("earth", "fuxi", "伏羲气象大模型，高精度短期气候预测"),
            "corrdiff": ("earth", "corrdiff", "CorrDiff，基于扩散模型的气候预测"),
            "fengwu": ("earth", "fengwu", "风乌气象大模型，多尺度气象预报"),
            "graphcast": ("earth", "graphcast", "GraphCast，基于图神经网络的天气预报"),
            "graphcast_jax": ("earth", "graphcast_jax", "GraphCast JAX版本"),
            "nowcastnet": ("earth", "nowcastnet", "NowCastNet，短临降水预报模型"),
            "oceancast": ("earth", "oceancast", "OceanCast，海洋环流预测模型"),
            "xihe": ("earth", "xihe", "羲和气象大模型，面向极端天气事件预测"),
            "deepcfd": ("cfd", "DeepCFD", "DeepCFD，基于深度学习的CFD模拟"),
            "cfd_benchmark": ("cfd", "CFD_Benchmark", "CFD基准测试套件"),
            "f_fno": ("cfd", "CFD_Benchmark", "F_FNO", "F-FNO，傅里叶神经算子变体"),
            "fno": ("cfd", "CFD_Benchmark", "FNO", "FNO，傅里叶神经算子"),
            "factformer": ("cfd", "CFD_Benchmark", "Factformer", "FactFormer，因子分解变换器"),
            "gfno": ("cfd", "CFD_Benchmark", "GFNO", "GFNO，几何傅里叶神经算子"),
            "gnot": ("cfd", "CFD_Benchmark", "GNOT", "GNOT，图神经算子"),
            "galerkin_transformer": ("cfd", "CFD_Benchmark", "Galerkin_Transformer", "Galerkin Transformer，伽辽金变换器"),
            "lsm": ("cfd", "CFD_Benchmark", "LSM", "LSM，线性稳定性模型"),
            "mwt": ("cfd", "CFD_Benchmark", "MWT", "MWT，多尺度小波变换器"),
            "ono": ("cfd", "CFD_Benchmark", "ONO", "ONO，算子神经算子"),
            "swin": ("cfd", "CFD_Benchmark", "Swin", "Swin Transformer，滑动窗口变换器"),
            "transformer": ("cfd", "CFD_Benchmark", "Transformer", "Transformer，标准变换器架构"),
            "transolver": ("cfd", "CFD_Benchmark", "Transolver", "Transolver，变换求解器"),
            "u_fno": ("cfd", "CFD_Benchmark", "U_FNO", "U-FNO，统一傅里叶神经算子"),
            "u_no": ("cfd", "CFD_Benchmark", "U_NO", "U-NO，统一神经算子"),
            "u_net": ("cfd", "CFD_Benchmark", "U_Net", "U-Net，经典图像分割网络"),
            "evo2": ("biosciences", "evo2", "Evo2，蛋白质进化语言模型"),
            "mace": ("matchem", "mace", "MACE，材料科学机器学习势"),
        }
        for alias, val in raw.items():
            domain, model_dir = val[0], val[1]
            desc = val[-1]
            sub_model = val[2] if len(val) >= 4 else ""
            self._aliases[alias] = (domain, model_dir, sub_model)
            self._descriptions[alias] = desc

    def resolve(self, name: str) -> t.Optional[dict]:
        name_lower = name.lower()
        if name_lower in self._aliases:
            domain, model_dir, sub_model = self._aliases[name_lower]
            return {
                "alias": name_lower,
                "domain": domain,
                "model": model_dir,
                "sub_model": sub_model,
                "description": self._descriptions.get(name_lower, ""),
            }
        found = self._discover_by_name(name_lower)
        if found:
            return found
        return None

    def _discover_by_name(self, name: str) -> t.Optional[dict]:
        if not EXAMPLES_DIR.exists():
            return None
        for domain_dir in EXAMPLES_DIR.iterdir():
            if not domain_dir.is_dir() or domain_dir.name == "configs":
                continue
            domain_key = DOMAIN_DIR_MAP.get(domain_dir.name)
            if not domain_key:
                continue
            for model_dir in domain_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name == "conf":
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
                    }
        return None

    def list_models(self, domain: t.Optional[str] = None) -> t.List[dict]:
        results = []
        seen = set()
        for alias, (domain_key, model_dir, sub_model) in self._aliases.items():
            if domain and domain_key != domain:
                continue
            key = f"{domain_key}|{model_dir}|{sub_model}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "alias": alias,
                    "model": model_dir,
                    "domain": domain_key,
                    "domain_desc": DOMAIN_DESCRIPTIONS.get(domain_key, ""),
                    "description": self._descriptions.get(alias, ""),
                    "sub_model": sub_model,
                })
        if not EXAMPLES_DIR.exists():
            return results
        for domain_dir in EXAMPLES_DIR.iterdir():
            if not domain_dir.is_dir() or domain_dir.name == "configs":
                continue
            domain_key = DOMAIN_DIR_MAP.get(domain_dir.name)
            if not domain_key or (domain and domain_key != domain):
                continue
            for model_dir in domain_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name == "conf":
                    continue
                auto_alias = model_dir.name.replace("_", "").lower()
                key = f"{domain_key}|{model_dir.name}|"
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "alias": auto_alias,
                        "model": model_dir.name,
                        "domain": domain_key,
                        "domain_desc": DOMAIN_DESCRIPTIONS.get(domain_key, ""),
                        "description": "自动发现的模型",
                        "sub_model": "",
                    })
        return results


class ModuleRegistry:
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


model_registry = ModelRegistry()
module_registry = ModuleRegistry()
