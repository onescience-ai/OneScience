"""
OneScience CLI 配置系统

支持:
  - 内置预设 (built-in preset, 无需文件)
  - 自定义配置 (.onescience.json, 项目根目录)
  - 环境切换 (env use)

模型发现优先级 (5 级):
  1. 完整路径 (含 / 或 \\)
  2. 当前环境的自定义模型 (config.models)
  3. 当前环境的扫描目录 (config.model_roots)
  4. 当前工作目录下的同名目录
  5. 内置预设 (built-in)
"""

import os
import json
import typing as t
from pathlib import Path

# =========================================================================
#  内置预设 (built-in preset)
# =========================================================================

BUILTIN_NAME = "onescience"
BUILTIN_DATASETS_DIR = "/public/share/sugonhpcapp01/onestore/onedatasets"

BUILTIN_DOMAINS: t.Dict[str, str] = {
    "earth": "气象与气候",
    "cfd": "计算流体力学",
    "biosciences": "生物科学",
    "matchem": "材料化学",
    "structural": "结构力学",
}

BUILTIN_DOMAIN_DIR_MAP: t.Dict[str, str] = {
    "earth": "earth",
    "cfd": "cfd",
    "biosciences": "biosciences",
    "bio": "biosciences",
    "matchem": "matchem",
    "MaterialsChemistry": "matchem",
    "structural": "structural",
}

BUILTIN_DATASETS: t.Dict[str, str] = {
    "airfoil": "CFD_Benchmark/airfoil",
    "darcy": "CFD_Benchmark/darcy",
    "elasticity": "CFD_Benchmark/elasticity",
    "ns": "CFD_Benchmark/ns",
    "pipe": "CFD_Benchmark/pipe",
    "plasticity": "CFD_Benchmark/plasticity",
    "era5": "ERA5",
    "era5_stats": "ERA5/stats",
    "era5_static": "ERA5/static",
    "cwb": "corrdiff/cwb",
    "graphcast": "graphcast",
    "mace": "mace",
    "evo2": "evo2",
    "protenix": "protenix",
    "openfold": "openfold",
    "matris": "matris",
    "deepcfd": "DeepCFD",
    "beno": "BENO",
    "pdenneval": "PDENNEval",
    "lagrangian_mgn": "Lagrangian_MGN",
    "topology": "GP_for_TO",
}

# (alias) -> (domain, model_dir, sub_model, description)
BUILTIN_MODELS_RAW: t.Dict[str, tuple] = {
    "pangu": ("earth", "pangu_weather", "", "盘古气象大模型"),
    "fourcastnet": ("earth", "fourcastnet", "", "FourCastNet"),
    "fuxi": ("earth", "fuxi", "", "伏羲气象大模型"),
    "corrdiff": ("earth", "corrdiff", "", "CorrDiff"),
    "fengwu": ("earth", "fengwu", "", "风乌气象大模型"),
    "graphcast": ("earth", "graphcast", "", "GraphCast"),
    "graphcast_jax": ("earth", "graphcast_jax", "", "GraphCast JAX"),
    "nowcastnet": ("earth", "nowcastnet", "", "NowCastNet"),
    "oceancast": ("earth", "oceancast", "", "OceanCast"),
    "xihe": ("earth", "xihe", "", "羲和气象大模型"),
    "deepcfd": ("cfd", "DeepCFD", "", "DeepCFD"),
    "cfd_benchmark": ("cfd", "CFD_Benchmark", "", "CFD基准测试"),
    "f_fno": ("cfd", "CFD_Benchmark", "F_FNO", "F-FNO"),
    "fno": ("cfd", "CFD_Benchmark", "FNO", "FNO"),
    "factformer": ("cfd", "CFD_Benchmark", "Factformer", "FactFormer"),
    "gfno": ("cfd", "CFD_Benchmark", "GFNO", "GFNO"),
    "gnot": ("cfd", "CFD_Benchmark", "GNOT", "GNOT"),
    "galerkin_transformer": ("cfd", "CFD_Benchmark", "Galerkin_Transformer", "Galerkin Transformer"),
    "lsm": ("cfd", "CFD_Benchmark", "LSM", "LSM"),
    "mwt": ("cfd", "CFD_Benchmark", "MWT", "MWT"),
    "ono": ("cfd", "CFD_Benchmark", "ONO", "ONO"),
    "swin": ("cfd", "CFD_Benchmark", "Swin", "Swin Transformer"),
    "transformer": ("cfd", "CFD_Benchmark", "Transformer", "Transformer"),
    "transolver": ("cfd", "CFD_Benchmark", "Transolver", "Transolver"),
    "u_fno": ("cfd", "CFD_Benchmark", "U_FNO", "U-FNO"),
    "u_no": ("cfd", "CFD_Benchmark", "U_NO", "U-NO"),
    "u_net": ("cfd", "CFD_Benchmark", "U_Net", "U-Net"),
    "evo2": ("biosciences", "evo2", "", "Evo2"),
    "mace": ("matchem", "mace", "", "MACE"),
}

# =========================================================================
#  配置加载
# =========================================================================


def _find_config_file() -> t.Optional[Path]:
    """从 cwd 向上查找 .onescience.json"""
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        cand = parent / ".onescience.json"
        if cand.exists():
            return cand
    return None


def _find_project_root_by_setup() -> t.Optional[Path]:
    """向上查找 setup.py + examples 目录（兼容旧的 onescience 项目）"""
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "setup.py").exists() and (parent / "examples").exists():
            return parent
    return None


class Config:
    """项目配置，合并内置预设 + 自定义配置"""

    def __init__(self):
        self._file_path: t.Optional[Path] = None
        self._data: t.Dict = {}

        # 自动加载
        config_file = _find_config_file()
        if config_file:
            self._load_file(config_file)

    def _load_file(self, path: Path):
        try:
            self._data = json.loads(path.read_text(encoding="utf-8"))
            self._file_path = path
        except (json.JSONDecodeError, OSError):
            self._data = {}

    # ---- 基本属性 ----

    @property
    def name(self) -> str:
        return self._data.get("name", BUILTIN_NAME)

    @property
    def file_path(self) -> t.Optional[Path]:
        return self._file_path

    # ---- 自定义模型 ----

    @property
    def custom_models(self) -> t.Dict[str, t.Dict]:
        """返回当前环境的自定义模型"""
        return self._data.get("models", {})

    @property
    def model_roots(self) -> t.List[Path]:
        """返回自定义扫描目录列表"""
        roots = self._data.get("model_roots", [])
        config_dir = self._file_path.parent if self._file_path else Path.cwd()
        result = []
        for r in roots:
            p = Path(r)
            if not p.is_absolute():
                p = (config_dir / p).resolve()
            result.append(p)
        return result

    # ---- 数据集 ----

    @property
    def datasets(self) -> t.Dict[str, str]:
        """返回当前环境的数据集映射（名称 → 路径）"""
        custom = self._data.get("datasets", {})
        merged = dict(BUILTIN_DATASETS)
        merged.update(custom)
        return merged

    @property
    def datasets_dir(self) -> str:
        return self._data.get("datasets_dir", BUILTIN_DATASETS_DIR)

    # ---- 领域 ----

    @property
    def domains(self) -> t.Dict[str, str]:
        custom = self._data.get("domains", {})
        merged = dict(BUILTIN_DOMAINS)
        merged.update(custom)
        return merged

    @property
    def domain_dir_map(self) -> t.Dict[str, str]:
        custom = self._data.get("domain_dir_map", {})
        merged = dict(BUILTIN_DOMAIN_DIR_MAP)
        merged.update(custom)
        return merged

    # ---- 模型发现（5 级优先级） ----

    def resolve_model(self, name: str) -> t.Optional[dict]:
        """5 级优先级查找模型"""
        name_lower = name.lower().strip()

        # 级别 1: 完整路径
        if "/" in name or "\\" in name:
            p = Path(name)
            if p.exists() and p.is_dir():
                return {
                    "alias": p.name,
                    "domain": "_custom",
                    "model": str(p),
                    "sub_model": "",
                    "description": f"路径模型: {p}",
                    "model_dir": p,
                    "source": "path",
                }

        # 级别 2: 当前环境的自定义模型
        custom = self.custom_models
        if name_lower in custom:
            meta = custom[name_lower]
            domain = meta.get("domain", "_custom")
            model_dir = Path(meta["dir"]) if "dir" in meta else None
            return {
                "alias": name_lower,
                "domain": domain,
                "model": model_dir.name if model_dir else name_lower,
                "sub_model": meta.get("sub_model", ""),
                "description": meta.get("description", "自定义模型"),
                "model_dir": model_dir,
                "source": "config",
            }

        # 级别 3: 扫描目录
        for root in self.model_roots:
            candidate = root / name_lower
            if candidate.exists() and candidate.is_dir():
                return {
                    "alias": name_lower,
                    "domain": "_custom",
                    "model": candidate.name,
                    "sub_model": "",
                    "description": f"扫描目录模型: {candidate}",
                    "model_dir": candidate,
                    "source": "scan",
                }

        # 级别 4: 当前工作目录
        cwd_candidate = Path.cwd() / name_lower
        if cwd_candidate.exists() and cwd_candidate.is_dir():
            return {
                "alias": name_lower,
                "domain": "_custom",
                "model": cwd_candidate.name,
                "sub_model": "",
                "description": f"当前目录模型: {cwd_candidate}",
                "model_dir": cwd_candidate,
                "source": "cwd",
            }

        # 级别 5: 内置预设
        if name_lower in BUILTIN_MODELS_RAW:
            domain, model_dir, sub_model, desc = BUILTIN_MODELS_RAW[name_lower]
            # 尝试定位内置模型的目录
            project_root = _find_project_root_by_setup()
            if project_root:
                model_path = project_root / "examples" / BUILTIN_DOMAIN_DIR_MAP.get(domain, domain) / model_dir
            else:
                model_path = None
            return {
                "alias": name_lower,
                "domain": domain,
                "model": model_dir,
                "sub_model": sub_model,
                "description": desc,
                "model_dir": model_path,
                "source": "builtin",
            }

        return None

    def list_models(self, domain: t.Optional[str] = None) -> t.List[dict]:
        """列出所有可用模型（内置 + 自定义 + 扫描目录发现）"""
        seen = set()
        results = []

        # 内置模型
        for alias, (domain_key, model_dir, sub_model, desc) in BUILTIN_MODELS_RAW.items():
            if domain and domain_key != domain:
                continue
            key = f"{domain_key}|{model_dir}|{sub_model}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "alias": alias,
                    "model": model_dir,
                    "domain": domain_key,
                    "domain_desc": self.domains.get(domain_key, ""),
                    "description": desc,
                    "sub_model": sub_model,
                    "source": "builtin",
                })

        # 自定义模型
        for alias, meta in self.custom_models.items():
            domain_key = meta.get("domain", "_custom")
            if domain and domain_key != domain:
                continue
            model_name = Path(meta.get("dir", alias)).name if "dir" in meta else alias
            key = f"{domain_key}|{model_name}|{meta.get('sub_model', '')}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "alias": alias,
                    "model": model_name,
                    "domain": domain_key,
                    "domain_desc": self.domains.get(domain_key, domain_key),
                    "description": meta.get("description", "自定义模型"),
                    "sub_model": meta.get("sub_model", ""),
                    "source": "config",
                })

        # 扫描目录发现的模型
        for root in self.model_roots:
            if not root.exists():
                continue
            for item in sorted(root.iterdir()):
                if not item.is_dir() or item.name.startswith("."):
                    continue
                key = f"_scan|{item.name}|"
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "alias": item.name.lower(),
                        "model": item.name,
                        "domain": "_custom",
                        "domain_desc": "自定义",
                        "description": f"扫描目录: {item}",
                        "sub_model": "",
                        "source": "scan",
                    })

        return results

    def resolve_dataset(self, name: str) -> t.Optional[str]:
        """解析数据集名称 → 完整路径

        优先级:
          1. 完整路径 (含 / 或 \\)
          2. 当前环境的自定义数据集
          3. 内置预设数据集
          4. datasets_dir 下的同名子目录
        """
        if "/" in name or "\\" in name:
            if os.path.exists(name):
                return os.path.abspath(name)
            return None

        # 自定义数据集
        custom = self._data.get("datasets", {})
        if name in custom:
            p = Path(custom[name])
            if not p.is_absolute():
                if self._file_path:
                    p = (self._file_path.parent / p).resolve()
            if p.exists():
                return str(p)
            return str(p)  # 即使不存在也返回路径（用于下载等场景）

        # 内置数据集
        if name in BUILTIN_DATASETS:
            p = Path(self.datasets_dir) / BUILTIN_DATASETS[name]
            if p.exists():
                return str(p)
            return str(p)

        # datasets_dir 下的同名子目录
        p = Path(self.datasets_dir) / name
        if p.exists():
            return str(p)

        return None


# 全局单例
config = Config()
