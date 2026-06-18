import re
import os
import importlib.util
from setuptools import setup, find_packages
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

# ==========================================
# 1. 工具函数
# ==========================================
# 全局缓存状态，防止重复扫描和重复执行
BUILD_HOOKS_EXECUTED = False

def parse_requirements(filename):
    """从 requirements.txt 解析依赖列表，跳过空行和注释行"""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def resolve(requires, deps_dict):
    """将短名称列表映射为 requirements.txt 中的完整版本约束"""
    return [deps_dict[r] for r in requires]


def unique(items):
    """列表去重，保持原始顺序"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def discover_package_data():
    """
    自动发现子模块的 package_data 配置
    扫描 src/onescience 下所有子模块，查找 package_config.py 文件
    """
    package_data = {}
    src_dir = os.path.join(os.path.dirname(__file__), "src", "onescience")

    for root, dirs, files in os.walk(src_dir):
        if "package_config.py" in files:
            try:
                rel_path = os.path.relpath(root, src_dir)
                module_parts = ["onescience"] + rel_path.split(os.sep) if rel_path != "." else ["onescience"]
                module_name = ".".join(module_parts)

                config_path = os.path.join(root, "package_config.py")
                spec = importlib.util.spec_from_file_location(f"{module_name}.package_config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)

                if hasattr(config_module, 'get_package_data'):
                    submodule_data = config_module.get_package_data()
                    package_data.update(submodule_data)
                    print(f"  Discovered package config from: {module_name}")

            except Exception as e:
                print(f"  Failed to load package config from {root}: {e}")

    return package_data

def discover_build_hooks():
    """收集所有子模块声明的构建钩子"""
    hooks = []
    src_dir = os.path.join(os.path.dirname(__file__), "src", "onescience")

    for root, dirs, files in os.walk(src_dir):
        if "package_config.py" in files:
            try:
                rel_path = os.path.relpath(root, src_dir)
                module_parts = ["onescience"] + rel_path.split(os.sep) if rel_path != "." else ["onescience"]
                module_name = ".".join(module_parts)

                config_path = os.path.join(root, "package_config.py")
                spec = importlib.util.spec_from_file_location(f"{module_name}.package_config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)

                if hasattr(config_module, "get_build_hook"):
                    hook = config_module.get_build_hook()
                    if hook is not None:
                        hooks.append(hook)

            except Exception as e:
                print(f"  Failed to load build hook from {root}: {e}")

    return hooks


def run_build_hooks():
    """
    执行所有构建钩子
    使用全局标记确保整个构建过程中只执行一次
    """
    global BUILD_HOOKS_EXECUTED
    if BUILD_HOOKS_EXECUTED:
        return

    project_root = os.path.dirname(__file__)
    env = os.environ.copy()
    src_dir = os.path.join(project_root, "src")
    env["PYTHONPATH"] = (
        src_dir + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("PYTHONPATH")
        else src_dir
    )

    hooks = discover_build_hooks()
    print(f"Running {len(hooks)} package build hook(s)")
    for hook in hooks:
        hook(
            project_root=project_root,
            env=env,
            python_executable=sys.executable,
            subprocess_module=subprocess,
        )

    BUILD_HOOKS_EXECUTED = True


# ==========================================
# 构建钩子自定义命令
# 在 build_py / develop 阶段自动触发所有构建钩子
# ==========================================

class build_py(_build_py):
    def run(self):
        run_build_hooks()
        super().run()


class develop(_develop):
    def run(self):
        run_build_hooks()
        super().run()
 
# ==========================================
# 2. 依赖配置区 (后续维护仅需修改此处)
# ==========================================
# 约定：
# - core_requires:        所有领域都依赖的基础包
# - earth/cfd/bio/matchem_requires: 特定领域的定制依赖

one_deps = parse_requirements("requirements.txt")
deps = {re.split(r"[=<>~!]", dep)[0]: dep for dep in one_deps}

core_requires = [
    "numpy",
    "tqdm",
    "treelib",
    "hydra-core",
    "termcolor",
    "wandb",
    "mlflow",
    "pyyaml",
    "h5py",
    "ruamel.yaml",
    "scikit-learn",
    "einops",
    "pandas",
    "omegaconf",
    "pybind11",
    "matplotlib",
    "pytz",
    "s3fs",
    "requests",
    "importlib_metadata",
    "scipy",
    "torchdata",
    "setuptools",
    "click",
]

earth_requires = [
    "timm",
    "xarray",
    "zarr",
    "netcdf4",
    "dask",
    "cftime",
    "seaborn",
    "opencv-python",
    "absl-py",
]

cfd_requires = [
    "timm",
    "vtk",
    "pyvista",
    "shapely",
    "torch_geometric",
    "deepxde",
    "gpytorch",
    "seaborn",
    "numba",
]

bio_requires = [
    #"megatron-core",
    "lmdb",
    "orjson",
    "ml-collections",
    "dm-tree",
    "dm-haiku",
    "diffrax",
    "biopandas",
    "biopython",
    "pyrsistent",
    "chex",
    "flax",
    "fiddle",
    "lightning",
    "sentencepiece",
    "datasets",
    "braceexpand",
    "webdataset",
    "nemo_run",
    "tiktoken",
    "zstandard",
    "transformers",
    "ftfy",
    "modelcif",
    "ihm",
    "mashumaro",
    "py3Dmol",
    "biotite",
    "rdkit",
    "p_tqdm",
    "gemmi",
    "hydra-colorlog",
    "fairscale",
    # alphagenome dependencies
    "alphagenome",
    "kagglehub",
    "orbax-checkpoint",
    "pyfaidx",
    "jaxtyping",
    "einshape",
    "filelock",
    "absl-py",
    "jmp",
    "ml-dtypes",
    "opt-einsum",
    "ninja",
    "contextlib2",
    "psutil",
    "optree",
    "gpytorch",
    "torch_geometric",
    "redis",
    "pillow",
    "tabulate",
    "typeguard",
    "pytest",
    "pdbfixer",
    "e3nn",
    "pyranges",
]

matchem_requires = [
    "ase",
    "pymatgen",
    "e3nn",
    "matscipy",
    "python-hostlist",
    "configargparse",
    "lmdb",
    "orjson",
    "ase_db_backends",
    "submitit",
    "clusterscope",
    "huggingface_hub",
    "numba",
    "opt_einsum-fx",
    "torchtnt",
    "torchmetrics",
    "torch-ema",
    "prettytable",
    'pytest',
    "cuequivariance",
    "pwdata",
    "scikit-learn-intelex",
    "pwact",
]

chemistry_requires = [
    "e3nn",
    "ase",
    "xtb",
    "rdkit",
    "matscipy",
    "python-hostlist",
    "configargparse",
    "lmdb",
    "orjson",
    "pymatgen",
    "ase_db_backends",
    "submitit",
    "clusterscope",
    "huggingface_hub",
    "numba",
    "opt_einsum-fx",
    "torchtnt",
]


# ==========================================
# 3. 构建与安装
# ==========================================

extras = {
    "earth":   resolve(unique(earth_requires), deps),
    "cfd":     resolve(unique(cfd_requires), deps),
    "bio":     resolve(unique(bio_requires), deps),
    "matchem": resolve(unique(matchem_requires), deps),
    "all":     resolve(unique(earth_requires + cfd_requires + bio_requires + matchem_requires), deps),
}

setup(
    name="onescience",
    version="0.3.0",
    author="sugon-ai4s",
    author_email="ai4s@sugon.com",
    description="First release",
    long_description="OneScience is a scientific computing toolkit built on an advanced deep learning framework",
    url="https://github.com/onescience-ai/OneScience",
    python_requires=">=3.10.0",
    package_dir={
        "": "src",
        "cli": "cli/cli",
        "cli.commands": "cli/cli/commands",
        "cli.core": "cli/cli/core",
    },
    packages=find_packages("src") + ["cli", "cli.commands", "cli.core"],
    install_requires=resolve(unique(core_requires), deps),
    extras_require=extras,
    include_package_data=True,
    package_data=discover_package_data(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "onescience=cli.main:cli",
        ],
    },
    cmdclass={
      "build_py": build_py,
      "develop": develop,
    },
)
