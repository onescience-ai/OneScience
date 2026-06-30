"""
OneScience CLI 配置系统

支持:
  - 内置预设 (built-in preset, 无需文件)
  - 自定义配置 (.onescience.json, 项目根目录)
  - 环境切换 (env use)

模型发现优先级 (6 级):
  1. 完整路径 (含 / 或 \\)
  2. 当前环境的自定义模型 (config.models)
  3. 当前环境的扫描目录 (config.model_roots)
  4. 当前工作目录下的同名目录
  5. 内置预设 (built-in)
  6. ModelScope 自动下载（本地不存在时从 ModelScope 拉取）
"""

import os
import json
import typing as t
from pathlib import Path
import click

# =========================================================================
#  内置预设 (built-in preset)
# =========================================================================

BUILTIN_NAME = "onescience"
if os.name == "nt":
    BUILTIN_DATASETS_DIR = str(Path.home() / ".onescience" / "datasets")
    BUILTIN_MODELS_DIR = str(Path.home() / ".onescience" / "models")
else:
    BUILTIN_DATASETS_DIR = "/public/share/sugonhpcapp01/onestore/onedatasets"
    BUILTIN_MODELS_DIR = "/public/share/sugonhpcapp01/onestore/onemodels"

# 状态缓存文件：记录不可写时自动回退的路径，实现"首次设置，后续直接使用"
_PATH_CACHE_FILE = Path.home() / ".onescience" / ".path_cache.json"


def _update_cache_entry(key: str, value: str) -> None:
    """更新缓存文件中的单条记录，保留已有其他记录"""
    cache = {}
    if _PATH_CACHE_FILE.exists():
        try:
            cache = json.loads(_PATH_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    cache[key] = value
    _PATH_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_writable_dir(key: str, default_path: str, fallback_base: str) -> str:
    """检测默认路径是否可写，不可写时自动回退并缓存结果

    优先级:
      1. 状态缓存文件（持久化上次决定）
      2. 尝试默认路径是否可写或已有数据（只读也可用）
      3. 回退到 ~/.onescience/{fallback_base}
    """
    cache_dir = _PATH_CACHE_FILE.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. 检查缓存
    if _PATH_CACHE_FILE.exists():
        try:
            cache = json.loads(_PATH_CACHE_FILE.read_text(encoding="utf-8"))
            if key in cache:
                return cache[key]
        except (json.JSONDecodeError, OSError):
            pass

    # 2. 检查默认路径是否已有数据（只读场景也使用）
    default = Path(default_path)
    if default.exists() and any(default.iterdir()):
        _update_cache_entry(key, default_path)
        return default_path

    # 3. 尝试默认路径是否可写
    try:
        default.mkdir(parents=True, exist_ok=True)
        # 可写，缓存并返回
        _update_cache_entry(key, default_path)
        return default_path
    except OSError:
        # 4. 不可写，回退到用户目录
        fallback = str(Path.home() / ".onescience" / fallback_base)
        Path(fallback).mkdir(parents=True, exist_ok=True)
        _update_cache_entry(key, fallback)
        return fallback

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
    "airfrans": "Transolver-Airfoil-Design/Dataset",
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
    "newdataset": "NewModel/data",    
}

# ModelScope 数据集名称映射（数据集名 → ModelScope 上的数据集名）
# 当本地不存在时，自动从 ModelScope 拉取
MODELSCOPE_DATASETS: t.Dict[str, str] = {
    "airfoil": "airfoil",
    "airfrans": "airfrans",
    "darcy": "darcy",
    "elasticity": "elasticity",
    "ns": "ns",
    "pipe": "pipe",
    "plasticity": "plasticity",
    "era5": "ERA5",
    "era5_stats": "ERA5",
    "era5_static": "ERA5",
    "cwb": "corrdiff-cwb",
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

# ModelScope 模型名称映射（模型别名 → ModelScope 上的模型仓库名）
# 当本地不存在时，自动从 ModelScope 拉取
# 地址: https://www.modelscope.cn/organization/OneScience?tab=model
MODELSCOPE_MODELS: t.Dict[str, str] = {
    "pangu": "pangu_weather",
    "fourcastnet": "fourcastnet",
    "fuxi": "fuxi",
    "corrdiff": "corrdiff",
    "fengwu": "fengwu",
    "graphcast": "graphcast",
    "graphcast_jax": "graphcast_jax",
    "nowcastnet": "nowcastnet",
    "oceancast": "oceancast",
    "xihe": "xihe",
    "deepcfd": "DeepCFD",
    "cfd_benchmark": "CFD_Benchmark",
    "f_fno": "CFD_Benchmark",
    "fno": "CFD_Benchmark",
    "factformer": "CFD_Benchmark",
    "gfno": "CFD_Benchmark",
    "gnot": "CFD_Benchmark",
    "galerkin_transformer": "CFD_Benchmark",
    "lsm": "CFD_Benchmark",
    "mwt": "CFD_Benchmark",
    "ono": "CFD_Benchmark",
    "swin": "CFD_Benchmark",
    "transformer": "CFD_Benchmark",
    "transolver": "CFD_Benchmark",
    "u_fno": "CFD_Benchmark",
    "u_no": "CFD_Benchmark",
    "u_net": "CFD_Benchmark",
    "evo2": "evo2",
    "mace": "mace",
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
#  ModelScope 下载提示
# =========================================================================


def _ensure_modelscope() -> bool:
    """检查 modelscope CLI 是否可用，不可用时自动安装（仅首次）"""
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "-m", "modelscope", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            raise FileNotFoundError
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "modelscope"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0:
                click.secho(f"modelscope 安装失败: {r.stderr.strip()[:200]}", fg="red")
                return False
            return True
        except Exception as e:
            click.secho(f"modelscope 安装异常: {e}", fg="red")
            return False


def _auto_download_dataset(name: str, target_dir: Path) -> t.Optional[str]:
    """数据集本地不存在时，自动从 ModelScope 拉取

    ModelScope 地址: https://www.modelscope.cn/organization/OneScience?tab=dataset
    使用 modelscope CLI 下载（自动处理 LFS），先检查是否已安装，未安装则自动安装。
    """
    import subprocess

    ms_name = MODELSCOPE_DATASETS.get(name, name)

    if target_dir.exists():
        return str(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if not _ensure_modelscope():
        return None

    click.secho(f"正在从 ModelScope 自动下载数据集 '{name}'...", fg="cyan")
    click.echo(f"  数据集: OneScience/{ms_name}")
    click.echo(f"  目标: {target_dir}")

    try:
        r = subprocess.run(
            ["modelscope", "download", "--dataset", f"OneScience/{ms_name}", "--local_dir", str(target_dir)],
            timeout=3600,
        )
        if r.returncode != 0:
            if target_dir.exists() and not any(target_dir.iterdir()):
                target_dir.rmdir()
            click.secho(f"数据集 '{name}' 从 ModelScope 下载失败（exit code {r.returncode}）", fg="red")
            return None
    except Exception as e:
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        click.secho(f"数据集 '{name}' 下载异常: {e}", fg="red")
        return None

    click.secho(f"数据集 '{name}' 下载完成", fg="green")
    return str(target_dir)


def _auto_download_model(name: str, target_dir: Path) -> t.Optional[Path]:
    """模型本地不存在时，自动从 ModelScope 拉取

    ModelScope 地址: https://www.modelscope.cn/organization/OneScience?tab=model
    使用 modelscope CLI 下载（自动处理 LFS），先检查是否已安装，未安装则自动安装。
    """
    import subprocess

    ms_name = MODELSCOPE_MODELS.get(name, name)

    if target_dir.exists():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if not _ensure_modelscope():
        return None

    click.secho(f"正在从 ModelScope 自动下载模型 '{name}'...", fg="cyan")
    click.echo(f"  模型: OneScience/{ms_name}")
    click.echo(f"  目标: {target_dir}")

    try:
        r = subprocess.run(
            ["modelscope", "download", "--model", f"OneScience/{ms_name}", "--local_dir", str(target_dir)],
            timeout=3600,
        )
        if r.returncode != 0:
            if target_dir.exists() and not any(target_dir.iterdir()):
                target_dir.rmdir()
            click.secho(f"模型 '{name}' 从 ModelScope 下载失败（exit code {r.returncode}）", fg="red")
            return None
    except Exception as e:
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        click.secho(f"模型 '{name}' 下载异常: {e}", fg="red")
        return None

    click.secho(f"模型 '{name}' 下载完成", fg="green")
    return target_dir


def _resolve_lfs_files(target_dir: Path, ms_name: str, resource_type: str = "datasets"):
    """检测目录中的 LFS 指针文件，并从 ModelScope 下载真实数据

    Args:
        target_dir: 下载目录
        ms_name: ModelScope 上的仓库名
        resource_type: "datasets" 或 "models"
    """
    import urllib.request

    base_url = f"https://www.modelscope.cn/{resource_type}/OneScience/{ms_name}/resolve/master"
    lfs_signature = "version https://git-lfs.github.com/spec/"
    lfs_failures = 0

    for fpath in target_dir.rglob("*"):
        if not fpath.is_file() or fpath.name == ".gitattributes":
            continue
        # 跳过已存在于 .git 目录中的文件
        if ".git" in str(fpath.relative_to(target_dir)).split(os.sep):
            continue
        try:
            # 只读取前 200 字节判断是否为 LFS 指针文件
            with open(fpath, "rb") as f:
                header = f.read(200).decode("utf-8", errors="ignore")
            if not header.startswith(lfs_signature):
                continue
            # 提取 oid
            for line in header.split("\n"):
                if line.startswith("oid sha256:"):
                    oid = line.split(":")[1].strip()
                    # 通过 HTTP 下载真实文件
                    file_rel = str(fpath.relative_to(target_dir)).replace("\\", "/")
                    file_url = f"{base_url}/{file_rel}"
                    try:
                        req = urllib.request.Request(file_url, headers={
                            "User-Agent": "Mozilla/5.0",
                            "Accept": "application/octet-stream",
                        })
                        with urllib.request.urlopen(req, timeout=3600) as resp:
                            real_data = resp.read()
                        fpath.write_bytes(real_data)
                    except Exception:
                        lfs_failures += 1
                    break
        except Exception:
            continue

    if lfs_failures > 0:
        click.secho(f"  警告: {lfs_failures} 个 LFS 文件下载失败，部分大文件可能不完整", fg="yellow")


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

    def reload(self):
        """重新加载配置文件（用于切换环境后更新配置）"""
        config_file = _find_config_file()
        if config_file:
            self._load_file(config_file)
        else:
            self._file_path = None
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
        env_dir = os.environ.get("ONESCIENCE_DATASETS_DIR")
        if env_dir:
            return env_dir
        config_dir = self._data.get("datasets_dir")
        if config_dir:
            return config_dir
        return _resolve_writable_dir("datasets_dir", BUILTIN_DATASETS_DIR, "datasets")

    @property
    def models_dir(self) -> str:
        """模型存储目录（用于 ModelScope 自动下载存放）"""
        env_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
        if env_dir:
            return env_dir
        config_dir = self._data.get("models_dir")
        if config_dir:
            return config_dir
        return _resolve_writable_dir("models_dir", BUILTIN_MODELS_DIR, "models")

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

    # ---- 模型发现（6 级优先级） ----

    def resolve_model(self, name: str) -> dict:
        """6 级优先级查找模型"""
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
            if model_path and model_path.exists():
                return {
                    "alias": name_lower,
                    "domain": domain,
                    "model": model_dir,
                    "sub_model": sub_model,
                    "description": desc,
                    "model_dir": model_path,
                    "source": "builtin",
                }

            # 级别 6: ModelScope 自动下载（内置模型本地不存在时）
            if name_lower in MODELSCOPE_MODELS:
                target = Path(self.models_dir) / model_dir
                result = _auto_download_model(name_lower, target)
                if result:
                    return {
                        "alias": name_lower,
                        "domain": domain,
                        "model": model_dir,
                        "sub_model": sub_model,
                        "description": f"从 ModelScope 自动下载: {desc}",
                        "model_dir": result,
                        "source": "modelscope",
                    }

            # 内置模型既不存在也无法下载
            return {
                "alias": name_lower,
                "domain": domain,
                "model": model_dir,
                "sub_model": sub_model,
                "description": desc,
                "model_dir": None,
                "source": "builtin",
            }

        # 级别 6: ModelScope 自动下载（所有未找到的模型）
        # 对于未注册的模型名，尝试从 ModelScope 拉取
        target = Path(self.models_dir) / name_lower
        result = _auto_download_model(name_lower, target)
        if result:
            return {
                "alias": name_lower,
                "domain": "_custom",
                "model": name_lower,
                "sub_model": "",
                "description": f"从 ModelScope 自动下载: {name_lower}",
                "model_dir": result,
                "source": "modelscope",
            }

        # 下载失败时返回带有 model_dir=None 的结果，而非 None
        # 这样 runner 可以展示更清晰的提示信息，而非"未知模型"
        return {
            "alias": name_lower,
            "domain": "_custom",
            "model": name_lower,
            "sub_model": "",
            "description": f"模型不存在且从 ModelScope 下载失败: {name_lower}",
            "model_dir": None,
            "source": "modelscope",
        }

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
          3. 内置预设数据集（本地不存在时从 ModelScope 自动下载）
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
            return None

        # 内置数据集 — 本地存在直接返回，不存在则自动从 ModelScope 拉取
        if name in BUILTIN_DATASETS:
            p = Path(self.datasets_dir) / BUILTIN_DATASETS[name]
            if p.exists() and any(p.iterdir()):
                return str(p)
            # fallback: 检查 data/{name} 子目录（兼容无软链接的目录结构）
            data_fallback = p.parent / "data" / name
            if data_fallback.exists() and any(data_fallback.iterdir()):
                return str(data_fallback)
            # 空目录则删除后重新下载
            if p.exists() and not any(p.iterdir()):
                p.rmdir()
            # 本地不存在，自动从 ModelScope 拉取
            rel = BUILTIN_DATASETS[name]
            if "/" in rel:
                # 子路径数据集（如 era5_stats → ERA5/stats），先确保父数据集已下载
                parent_rel = rel.split("/")[0]
                parent_key = next((k for k, v in BUILTIN_DATASETS.items() if v == parent_rel), name)
                parent_target = Path(self.datasets_dir) / parent_rel
                if not parent_target.exists() or not any(parent_target.iterdir()):
                    _auto_download_dataset(parent_key, parent_target)
                # 子路径数据集，下载父数据集后检查子路径是否存在
                return str(p) if p.exists() and any(p.iterdir()) else None
            # 非子路径数据集，直接下载
            target = p
            result = _auto_download_dataset(name, target)
            if result:
                return result
            return None

        # datasets_dir 下的同名子目录
        p = Path(self.datasets_dir) / name
        if p.exists() and any(p.iterdir()):
            return str(p)
        # 空目录则删除后重新触发下载
        if p.exists() and not any(p.iterdir()):
            p.rmdir()

        # 未知数据集，尝试从 ModelScope 自动拉取
        target = Path(self.datasets_dir) / name
        result = _auto_download_dataset(name, target)
        if result:
            return result

        return None


# 全局单例
config = Config()
