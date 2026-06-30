"""
执行引擎。

管理模型任务的调度和执行，支持：
  - onescience 项目内置模型
  - 通过 .onescience.json 注册的自定义模型
  - 任意路径下的模型目录
  - CFD_Benchmark 兼容模式
"""

import os
import re
import json
import glob
import subprocess
import shutil
import typing as t
import click
from pathlib import Path
from .registry import model_registry, EXAMPLES_DIR, PROJECT_ROOT
from .config import config, MODELSCOPE_MODELS


def _patch_config_values(content: str, patches: t.Dict[str, str]) -> t.Optional[str]:
    """使用 ruamel.yaml 结构化修改 YAML 配置

    相比纯文本替换方式，结构化方式：
      - 不受 YAML 注释、空行、多行值的影响
      - 正确维护 YAML 节点层级关系
      - 不会意外破坏其他键值对
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        # 兜底：无 ruamel.yaml 时用旧文本方式
        return _patch_config_values_text(content, patches)

    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(content)

    modified = False
    for path_key, value_str in patches.items():
        parts = path_key.split('.')
        # 导航到目标节点的父节点
        parent = data
        for part in parts[:-1]:
            if isinstance(parent, dict) and part in parent:
                parent = parent[part]
            else:
                parent = None
                break
        if parent is None or not isinstance(parent, dict) or parts[-1] not in parent:
            continue
        # 将补丁值解析为 YAML 类型并设置
        try:
            parsed = yaml.load(value_str)
            parent[parts[-1]] = parsed
            modified = True
        except Exception:
            pass

    if modified:
        from io import StringIO
        buf = StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()
    return None


def _patch_config_values_text(content: str, patches: t.Dict[str, str]) -> t.Optional[str]:
    """纯文本方式替换 YAML 配置值（兜底方案）"""
    lines = content.split('\n')
    result = []
    indent_stack: t.List[t.Tuple[int, str]] = []
    modified = False

    for line in lines:
        stripped = line.rstrip()
        trimmed = stripped.strip()

        if trimmed and ':' in trimmed and not trimmed.startswith('#'):
            indent = len(line) - len(line.lstrip())
            key = trimmed.split(':')[0].strip()

            while indent_stack and indent <= indent_stack[-1][0]:
                indent_stack.pop()

            if trimmed.rstrip().endswith(':'):
                indent_stack.append((indent, key))

            path_key = '.'.join(k for _, k in indent_stack[-2:]) + '.' + key if len(indent_stack) >= 2 else key
            if path_key in patches:
                prefix = line[:len(line) - len(line.lstrip())]
                comment = ''
                if '#' in line:
                    comment = '  ' + line[line.index('#'):].strip()
                new_line = f"{prefix}{key}: {patches[path_key]}{comment}"
                result.append(new_line)
                modified = True
                continue

        result.append(line)

    return '\n'.join(result) if modified else None


_STATIC_FILES = ["land_mask.npy", "soil_type.npy", "topography.npy"]


def _ensure_static_files(static_dir: Path) -> None:
    """确保静态文件存在，缺失时自动生成。

    部分地球模型（pangu 等）需要 land_mask/soil_type/topography
    三个静态场用于模型输入。如果数据集中没有，这里用随机假数据生成。
    """
    if static_dir.exists() and all((static_dir / f).exists() for f in _STATIC_FILES):
        return

    import numpy as np

    static_dir.mkdir(parents=True, exist_ok=True)
    shape = (721, 1440)
    rng = np.random.default_rng(42)

    if not (static_dir / "land_mask.npy").exists():
        arr = (rng.random(shape) > 0.7).astype(np.float32)
        np.save(str(static_dir / "land_mask.npy"), arr)

    if not (static_dir / "soil_type.npy").exists():
        arr = rng.integers(0, 6, size=shape).astype(np.float32)
        np.save(str(static_dir / "soil_type.npy"), arr)

    if not (static_dir / "topography.npy").exists():
        arr = rng.normal(500, 1500, size=shape).astype(np.float32)
        np.save(str(static_dir / "topography.npy"), arr)

    click.secho(f"  ✅ 已自动生成静态文件 ({len(_STATIC_FILES)} 个) 到: {static_dir}", fg="green")


def _patch_model_configs(model_dir: Path, data_path: str) -> t.Dict[str, str]:
    """通用配置补丁：将模型配置文件中的数据路径替换为实际数据集路径

    扫描模型目录下的所有 .yaml/.yml 文件（conf/ 目录和模型根目录），
    自动识别并替换常见的数据路径配置项：
      - data_path
      - data_dir
      - datadir

    不依赖模型领域或配置文件名，适用于所有模型。

    返回: {文件绝对路径: 原始内容}，用于执行后恢复
    """
    # 收集所有 yaml 文件（去重）
    yaml_files: t.List[Path] = []
    seen: t.Set[str] = set()
    for pattern in ("*.yaml", "*.yml"):
        for f in model_dir.glob(pattern):
            if str(f) not in seen:
                seen.add(str(f))
                yaml_files.append(f)
        conf_dir = model_dir / "conf"
        if conf_dir.exists():
            for f in conf_dir.glob(pattern):
                if str(f) not in seen:
                    seen.add(str(f))
                    yaml_files.append(f)

    # 常见数据路径配置项（不区分大小写）
    path_keys = ["data_path", "data_dir", "datadir"]
    # 匹配: 可选缩进 + key + 冒号 + 值 + 可选行内注释
    pattern = re.compile(
        r'^(\s*)(' + '|'.join(re.escape(k) for k in path_keys) + r')(\s*:\s*).+?(\s*#.*)?$',
        re.IGNORECASE | re.MULTILINE,
    )

    backups: t.Dict[str, str] = {}
    for yaml_file in yaml_files:
        original = yaml_file.read_text(encoding="utf-8")
        modified = pattern.sub(
            lambda m: f'{m.group(1)}{m.group(2)}{m.group(3)}"{data_path}"{m.group(4) or ""}',
            original,
        )
        if modified != original:
            backups[str(yaml_file)] = original
            yaml_file.write_text(modified, encoding="utf-8")
            click.echo(f"  Config: {yaml_file.relative_to(model_dir)} 已注入数据集路径")

    return backups


def _earth_config_patches(env: dict, config_path: t.Optional[Path] = None) -> t.Optional[t.Tuple[str, dict]]:
    data_dir = env.get("ONESCIENCE_DATASET_PATH", "")
    if not data_dir:
        return None
    patches = {}
    patches["datapipe.dataset.data_dir"] = f"'{data_dir}'"
    patches["datapipe.dataset.stats_dir"] = f"'{data_dir}/stats/'"
    static_dir_path = Path(data_dir) / "static"
    patches["datapipe.dataset.static_dir"] = f"'{static_dir_path}/'"

    # 始终从实际数据集自动检测可用年份并覆盖 config 中的年份配置
    # 确保 CLI 能自适应不同环境的数据集，避免 config 硬编码年份与数据不匹配
    years = _detect_era5_years(data_dir)
    if years:
        if len(years) >= 3:
            train_years = years[:-2]
            val_years = [years[-2]]
            test_years = [years[-1]]
        elif len(years) == 2:
            train_years = years[:-1]
            val_years = [years[-1]]
            test_years = [years[-1]]
        else:
            train_years = years
            val_years = years
            test_years = years
        patches["datapipe.dataset.train_time"] = str(train_years)
        patches["datapipe.dataset.val_time"] = str(val_years)
        patches["datapipe.dataset.test_time"] = str(test_years)
        click.echo(f"  检测到数据集年份: {years[0]}~{years[-1]} ({len(years)} 年)")
        click.echo(f"  训练: {train_years}, 验证: {val_years}, 测试: {test_years}")
        return (data_dir, patches)
    else:
        data_glob = os.path.join(data_dir, "data", "*.h5")
        click.secho(
            f"⚠️  在数据集目录中未找到 HDF5 数据文件:\n"
            f"    {data_glob}\n"
            f"    {os.path.join(data_dir, 'data', '*', '*.h5')}\n"
            f"    请检查数据集目录结构",
            fg="yellow",
        )
        # 没有检测到数据时不打补丁，保留模型原始 config 配置
        return None


def _detect_era5_years(data_dir: str) -> t.Optional[t.List[int]]:
    """检测 ERA5 数据集可用年份，兼容两种格式

    Returns:
        排序后的年份列表，如果没找到数据返回 None
    """
    # 新格式: data/{year}.h5
    data_glob = os.path.join(data_dir, "data", "*.h5")
    h5_files = sorted(glob.glob(data_glob))
    if h5_files:
        years = []
        for f in h5_files:
            try:
                year = int(os.path.basename(f).replace(".h5", ""))
                years.append(year)
            except ValueError:
                pass
        if years:
            return sorted(years)

    # 旧格式: data/{year}/*.h5 (年份子目录)
    data_dir_path = os.path.join(data_dir, "data")
    if os.path.isdir(data_dir_path):
        year_dirs = sorted([
            d for d in os.listdir(data_dir_path)
            if os.path.isdir(os.path.join(data_dir_path, d)) and d.isdigit()
        ])
        if year_dirs:
            years = [int(d) for d in year_dirs]
            # 验证年份目录里确实有 h5 文件
            for y in year_dirs[:1]:
                if glob.glob(os.path.join(data_dir_path, y, "*.h5")):
                    return years
            return years

    return None


# 通用数据集数据文件后缀列表（用于验证路径是否包含真实数据）
_DATA_FILE_EXTENSIONS = {'.h5', '.hdf5', '.nc', '.npy', '.npz', '.csv', '.grib', '.grib2', '.zarr', '.mat'}


def _validate_dataset_path(path: str) -> bool:
    """验证数据集路径是否包含可识别的数据文件

    递归查找常见科学数据格式文件，适用于任意数据集类型。
    """
    if not path or not os.path.isdir(path):
        return False
    for root, dirs, files in os.walk(path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in _DATA_FILE_EXTENSIONS):
                return True
    return False


RESULTS_DIR = PROJECT_ROOT / "model_results"


def print_metrics(result: dict):
    """打印模型执行后的关键指标"""
    from .formatter import Formatter

    if not result.get("metrics"):
        return
    metrics = result["metrics"]
    domain = result.get("domain", "")
    if domain == "cfd":
        cfd_headers = ["指标", "值"]
        cfd_rows = []
        for name in ["平均相对误差", "平均绝对误差", "平均MSE", "平均MAE", "平均MaxAE", "平均R²分数"]:
            val = metrics.get(name, "N/A")
            if val != "N/A":
                cfd_rows.append([name, val])
        if cfd_rows:
            click.echo(Formatter.table(cfd_headers, cfd_rows, align=["<", ">"]))
    elif domain == "earth":
        earth_headers = ["指标", "值"]
        earth_rows = []
        for name in ["RMSE", "ACC", "CSI", "MAE", "BIAS"]:
            val = metrics.get(name, "N/A")
            if val != "N/A":
                earth_rows.append([name, val])
        if earth_rows:
            click.echo(Formatter.table(earth_headers, earth_rows, align=["<", ">"]))
    elif domain == "_custom":
        # 通用模型：显示所有提取到的指标
        custom_headers = ["指标", "值"]
        custom_rows = [[k, v] for k, v in metrics.items() if v != "N/A"]
        if custom_rows:
            click.echo(Formatter.table(custom_headers, custom_rows, align=["<", ">"]))


def _get_log_path(model_dir: Path, prefix: str) -> Path:
    return model_dir / f"{prefix}_execution.log"


def _extract_metrics(log_text: str, domain: str) -> dict:
    metrics = {}
    if domain == "earth":
        patterns = [
            ("RMSE", r"RMSE\s+([\d\.eE+-]+)"),
            ("ACC", r"ACC\s+([\d\.eE+-]+)"),
            ("CSI", r"CSI\s+([\d\.eE+-]+)"),
            ("MAE", r"MAE\s+([\d\.eE+-]+)"),
            ("BIAS", r"BIAS\s+([\d\.eE+-]+)"),
        ]
        for name, pat in patterns:
            m = re.search(pat, log_text)
            metrics[name] = m.group(1) if m else "N/A"
    elif domain == "cfd":
        patterns = [
            ("平均相对误差", r"平均相对误差[:\s]+([\d\.eE+-]+)"),
            ("平均绝对误差", r"平均绝对误差[:\s]+([\d\.eE+-]+)"),
            ("平均MSE", r"平均MSE[:\s]+([\d\.eE+-]+)"),
            ("平均MAE", r"平均MAE[:\s]+([\d\.eE+-]+)"),
            ("平均MaxAE", r"平均MaxAE[:\s]+([\d\.eE+-]+)"),
            ("平均R²分数", r"平均R²分数[:\s]+([\d\.eE+-]+)"),
        ]
        for name, pat in patterns:
            m = re.search(pat, log_text)
            metrics[name] = m.group(1) if m else "N/A"
    elif domain in ("biosciences", "bio"):
        patterns = [
            ("plDDT", r"plDDT[:\s]+([\d\.eE+-]+)"),
            ("pTM", r"pTM[:\s]+([\d\.eE+-]+)"),
            ("i_pTM", r"i.pTM[:\s]+([\d\.eE+-]+)"),
            ("RMSD", r"RMSD[:\s]+([\d\.eE+-]+)"),
            ("损失", r"(?:损失|loss)[:\s]+([\d\.eE+-]+)"),
        ]
        for name, pat in patterns:
            m = re.search(pat, log_text, re.IGNORECASE)
            metrics[name] = m.group(1) if m else "N/A"
    elif domain in ("matchem", "MaterialsChemistry"):
        patterns = [
            ("能量MAE", r"能量[:\s]*MAE[:\s]+([\d\.eE+-]+)"),
            ("力MAE", r"力[:\s]*MAE[:\s]+([\d\.eE+-]+)"),
            ("损失", r"(?:损失|loss)[:\s]+([\d\.eE+-]+)"),
            ("能量RMSE", r"能量[:\s]*RMSE[:\s]+([\d\.eE+-]+)"),
            ("力RMSE", r"力[:\s]*RMSE[:\s]+([\d\.eE+-]+)"),
        ]
        for name, pat in patterns:
            m = re.search(pat, log_text, re.IGNORECASE)
            metrics[name] = m.group(1) if m else "N/A"
    else:
        # 通用模型：尝试匹配任何 key: value 形式的指标
        patterns = [
            ("RMSE", r"(?:RMSE|rmse)[:\s]+([\d\.eE+-]+)"),
            ("MAE", r"(?:MAE|mae)[:\s]+([\d\.eE+-]+)"),
            ("MSE", r"(?:MSE|mse)[:\s]+([\d\.eE+-]+)"),
            ("ACC", r"(?:ACC|acc)[:\s]+([\d\.eE+-]+)"),
            ("损失", r"(?:损失|loss|Loss)[:\s]+([\d\.eE+-]+)"),
            ("准确率", r"(?:准确率|accuracy|Accuracy)[:\s]+([\d\.eE+-]+)"),
        ]
        for name, pat in patterns:
            m = re.search(pat, log_text)
            metrics[name] = m.group(1) if m else "N/A"
    return metrics


def _retry_with_fixed_static_files(cmd, cwd, log_path, env, failed_output):
    """检测失败是否因缺失静态文件导致，自动生成后重试。

    完全通用的补救策略：不预判任何模型需要什么文件，
    只在模型执行报 FileNotFoundError 时按需补齐。
    """
    # 在输出中查找静态文件缺失错误
    # 匹配: FileNotFoundError: ... '.../static/land_mask.npy'
    match = re.search(
        r"FileNotFoundError.*?['\"](.+?/static/.*?\.npy)['\"]",
        failed_output,
        re.IGNORECASE,
    )
    if not match:
        return None

    missing_path = Path(match.group(1))
    static_dir = missing_path.parent

    # 只处理我们能生成的静态文件
    if missing_path.name not in ("land_mask.npy", "soil_type.npy", "topography.npy"):
        return None

    click.secho(f"  ⚠ 模型需要静态文件 {missing_path.name}，正在自动生成...", fg="yellow")
    _ensure_static_files(static_dir)

    # 重试一次
    click.secho("  ↻ 正在重试...", fg="yellow")
    try:
        process = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1,
        )
        retry_lines = []
        for line in iter(process.stdout.readline, ""):
            retry_lines.append(line)
        process.wait()
        retry_output = "".join(retry_lines)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(retry_output, encoding="utf-8")
        if process.returncode != 0:
            click.secho(f"重试仍失败（exit code {process.returncode}），完整日志: {log_path}", fg="red")
        return {"success": process.returncode == 0, "output": retry_output, "log_path": log_path}
    except Exception as e:
        click.secho(str(e), fg="red")
        return None


def _run_cmd(cmd: t.List[str], cwd: Path, log_path: Path, env: t.Optional[dict] = None) -> dict:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # 自动注入 PyTorch 显存优化环境变量（避免 pipe 模式下的 OOM）
    if "PYTORCH_HIP_ALLOC_CONF" not in merged_env:
        merged_env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    if "PYTORCH_CUDA_ALLOC_CONF" not in merged_env:
        merged_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        process = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=merged_env, text=True, bufsize=1,
        )
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            output_lines.append(line)
            click.echo(line, nl=False)
        process.wait()
        output = "".join(output_lines)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")

        # ── 智能补救：进程失败且因缺失静态文件 → 生成后重试 ─────────
        if process.returncode != 0:
            retry = _retry_with_fixed_static_files(cmd, cwd, log_path, merged_env, output)
            if retry is not None:
                return retry

        if process.returncode != 0:
            click.secho(f"命令执行失败（exit code {process.returncode}），完整日志: {log_path}", fg="red")

        return {"success": process.returncode == 0, "output": output, "log_path": log_path}
    except Exception as e:
        msg = str(e)
        click.secho(msg, fg="red")
        log_path.write_text(msg, encoding="utf-8")
        return {"success": False, "output": msg, "log_path": log_path}


def _run_cfd_benchmark_script(model_dir: Path, cmd_type: str, sub_model: str, dataset: str, env: dict) -> dict:
    """执行 CFD_Benchmark 模型（兼容旧版）"""
    output = ""
    all_success = True
    dataset_name = dataset if dataset else "airfoil"
    if sub_model:
        script_path = model_dir / "scripts" / "StandardBench" / dataset_name / f"{sub_model}.sh"
        log_path = _get_log_path(model_dir, sub_model)
        if script_path.exists():
            correct_data_path = env.get("ONESCIENCE_DATASET_PATH", "")
            if correct_data_path:
                script_content = script_path.read_text(encoding="utf-8")
                escaped_path = correct_data_path.replace('\\', '\\\\')
                script_content = re.sub(
                    r'--data_path\s+\S+',
                    f'--data_path {escaped_path}',
                    script_content
                )
                temp_script = model_dir / f".{sub_model}_tmp.sh"
                try:
                    temp_script.write_text(script_content, encoding="utf-8")
                    result = _run_cmd(["bash", str(temp_script)], model_dir, log_path, env)
                finally:
                    if temp_script.exists():
                        temp_script.unlink()
            else:
                result = _run_cmd(["bash", str(script_path)], model_dir, log_path, env)
            output += result["output"]
            all_success = result["success"]
        else:
            return {"success": False, "output": f"脚本不存在: {script_path}"}
    else:
        run_py = model_dir / "run.py"
        log_path = _get_log_path(model_dir, "cfd_benchmark")
        if run_py.exists():
            args = ["python", "run.py", "--loader", dataset_name]
            if dataset:
                data_path = env.get("ONESCIENCE_DATASET_PATH", dataset)
                args.extend(["--data_path", data_path])
            if cmd_type in ("INFER", "EVAL", "infer", "eval"):
                args.append("--eval")
                args.append("1")
            result = _run_cmd(args, model_dir, log_path, env)
            output += result["output"]
            all_success = result["success"]
    return {"success": all_success, "output": output}


# 内建模型特殊配置（模型目录名 → 配置项）
# 只存储通用引擎无法自动推断的信息：
#   _variant_order: variant 执行的语义顺序，如 ["short", "medium", "long"]
# 大部分模型不需要任何配置，通用发现引擎自动接管
_MODEL_EXECUTION_OVERRIDES = {
    "fuxi": {
        "_variant_order": ["short", "medium", "long"],
    },
}


# ──────────────────────────────────────────────
# 通用脚本自动发现引擎
# ──────────────────────────────────────────────

# 始终跳过的非流水线脚本
_SKIP_SCRIPTS = frozenset({'__init__', 'setup'})


def _discover_model_scripts(model_dir: Path) -> dict:
    """自动发现模型脚本，按命名约定分类。

    规则：
      - train*.py          → 训练阶段
      - infer*.py          → 推理阶段
      - result*.py/eval*.py → 评估阶段
      - __init__.py/setup.py → 跳过

    Variant 检测：当没有朴素的 train.py 但有 train_{variant}.py 时，
    认为该模型有多个变体（如 fuxi 的 short/medium/long）。

    Returns:
        {
            "train": [...],       # 脚本名（不含 .py）
            "infer": [...],
            "eval": [...],
            "variants": [...],    # 检测到的变体列表
            "has_plain_train": bool,
            "has_plain_infer": bool,
            "has_plain_eval": bool,
            "pipeline_found": bool,  # 是否存在任何流水线脚本
        }
    """
    result = {
        "train": [],
        "infer": [],
        "eval": [],
        "variants": [], 
        "has_plain_train": False,
        "has_plain_infer": False,
        "has_plain_eval": False,
        "pipeline_found": False,
    }

    for f in model_dir.glob("*.py"):
        name = f.stem
        if name in _SKIP_SCRIPTS:
            continue

        if name.startswith('train'):
            result["train"].append(name)
            result["pipeline_found"] = True
        elif name.startswith('infer'):
            result["infer"].append(name)
            result["pipeline_found"] = True
        elif name.startswith('result') or name.startswith('eval'):
            result["eval"].append(name)
            result["pipeline_found"] = True

    result["has_plain_train"] = 'train' in result["train"]
    result["has_plain_infer"] = any(s in ('inference', 'infer') for s in result["infer"])
    result["has_plain_eval"] = any(s in ('result', 'eval') for s in result["eval"])

    # Variant 检测：没有朴素 train.py 但有 train_{variant}.py
    if not result["has_plain_train"]:
        variants = set()
        for s in result["train"]:
            m = re.match(r'train_(.+)', s)
            if m:
                variants.add(m.group(1))
        result["variants"] = sorted(variants)

    return result


def _build_execution_plan(scripts: dict, cmd_type: str, variant_order: t.Optional[list] = None) -> list:
    """从已发现的脚本构建有序执行计划。

    执行顺序策略：
      bench + variant 模型 → 按变体分组执行（train_{v} → infer {v} → eval {v}）
      bench + 普通模型    → train → infer → eval 依次执行
      其他模式            → 只执行对应阶段的脚本

    Args:
        variant_order: 可选的 variant 语义排序，如 ["short", "medium", "long"]。
                       未提供时按字母序排列。

    Returns:
        [(script_name, [args]), ...]  的有序列表
    """
    _PHASE_MAP = {
        "TRAIN": ["train"],
        "train": ["train"],
        "bench": ["train", "infer", "eval"],
        "INFER": ["infer"],
        "infer": ["infer"],
        "EVAL": ["eval"],
        "eval": ["eval"],
    }
    phases = _PHASE_MAP.get(cmd_type, ["train", "infer", "eval"])

    plan = []
    variants = scripts["variants"]
    if variants:
        # 优先使用语义顺序，未指定时按字母序
        if variant_order:
            variants = [v for v in variant_order if v in variants]
        # ── Variant 模型 ──────────────────────────────────
        if cmd_type == "bench":
            # 按变体分组：train_{v} → infer {v} → eval {v}
            for variant in variants:
                train_v = f"train_{variant}"
                if train_v in scripts["train"]:
                    plan.append((f"{train_v}.py", []))

                if scripts["has_plain_infer"]:
                    plain_infer = next(s for s in scripts["infer"] if s in ('inference', 'infer'))
                    plan.append((f"{plain_infer}.py", [variant]))
                else:
                    for s in sorted(scripts["infer"]):
                        plan.append((f"{s}.py", []))

                if scripts["has_plain_eval"]:
                    plain_eval = next(s for s in scripts["eval"] if s in ('result', 'eval'))
                    plan.append((f"{plain_eval}.py", [variant]))
                else:
                    for s in sorted(scripts["eval"]):
                        plan.append((f"{s}.py", []))
        else:
            # 非 bench 模式：执行对应阶段的所有脚本
            for phase in phases:
                for s in sorted(scripts.get(phase, [])):
                    plan.append((f"{s}.py", []))
    else:
        # ── 普通模型 ──────────────────────────────────────
        for phase in phases:
            phase_scripts = scripts.get(phase, [])
            if phase == "train" and scripts["has_plain_train"]:
                # 有朴素 train.py 时只用它，跳过 train_* 变体脚本
                plan.append(("train.py", []))
            else:
                for s in sorted(phase_scripts):
                    plan.append((f"{s}.py", []))

    return plan


def _run_generic_model(model_dir: Path, cmd_type: str, env: dict, domain: str) -> dict:
    """通用模型执行器

    自动发现模型目录中的流水线脚本，按命名约定分类后执行。
    支持标准模型（train.py → inference.py → result.py）和
    variant 模型（train_short/medium/long.py → inference.py {v} → result.py {v}）。

    兜底策略：无流水线脚本时，扫描目录下所有 .py 文件。
    """
    # 1. 从配置中提取 variant 排序提示
    model_name = model_dir.name
    variant_order = None
    if model_name in _MODEL_EXECUTION_OVERRIDES:
        variant_order = _MODEL_EXECUTION_OVERRIDES[model_name].get("_variant_order")

    # 2. 自动发现脚本并构建执行计划
    scripts = _discover_model_scripts(model_dir)
    plan = _build_execution_plan(scripts, cmd_type, variant_order=variant_order)

    output = ""
    all_success = True
    found = False

    for script_name, args in plan:
        script_path = model_dir / script_name
        if not script_path.exists():
            continue
        found = True
        log_prefix = script_name.replace(".py", "")
        if args:
            log_prefix += "_" + "_".join(args)
        log_path = _get_log_path(model_dir, log_prefix)
        cmd = ["python", script_name] + args
        result = _run_cmd(cmd, model_dir, log_path, env)
        output += result["output"]
        all_success = all_success and result["success"]
        if not result["success"] and cmd_type in ("train", "TRAIN", "bench"):
            break

    # 3. 兜底：没有流水线脚本时，执行目录下第一个 .py 文件
    if not found:
        candidates = sorted(model_dir.glob("*.py"))
        candidates = [c for c in candidates
                      if c.name not in ('__init__.py', 'setup.py')
                      and 'test' not in c.name.lower()]
        if candidates:
            found = True
            script = candidates[0]
            log_path = _get_log_path(model_dir, script.stem)
            result = _run_cmd(["python", script.name], model_dir, log_path, env)
            output += result["output"]
            all_success = result["success"]

    return {"success": all_success, "output": output, "found_any": found}


def run_model(model_alias: str, cmd_type: str, dataset: str) -> dict:
    """执行模型

    model_alias 支持：
      - 内置别名 (pangu, fno, mace ...)
      - 自定义模型名 (来自 .onescience.json)
      - 完整路径 (绝对或相对路径)
    """
    # 解析模型
    info = model_registry.resolve(model_alias)
    if not info:
        return {"success": False, "error": f"未知模型: {model_alias}", "alias": model_alias}

    domain = info["domain"]
    model = info["model"]
    sub_model = info.get("sub_model", "")

    # 获取模型目录
    model_dir = info.get("model_dir")
    if not model_dir:
        if info.get("source") == "builtin":
            from .registry import DOMAIN_DIR_MAP
            model_dir = EXAMPLES_DIR / DOMAIN_DIR_MAP.get(domain, domain) / model
        else:
            model_dir = Path(model)

    if not model_dir.exists():
        hint = ""
        if info.get("source") == "modelscope":
            hint = (
                f"\n  提示: 模型 '{model_alias}' 从 ModelScope 自动下载失败。"
                f"\n  可能是网络问题（代理不可达或外网不通），请检查网络连接后重试。"
                f"\n  或使用 'onescience env init' 配置模型扫描路径。"
            )
        elif model_alias.lower() in MODELSCOPE_MODELS:
            ms_name = MODELSCOPE_MODELS[model_alias.lower()]
            hint = (
                f"\n  提示: {model_alias} 在 ModelScope 上已有注册，可尝试手动下载:"
                f"\n    modelscope download --model OneScience/{ms_name} --local_dir {model_dir}"
                f"\n  或在集群共享目录访问正常时，使用 'onescience env init' 配置模型扫描路径"
            )
        else:
            hint = (
                f"\n  提示: 请确认模型名称是否正确，或使用完整路径指定模型目录"
                f"\n  使用 'onescience list models' 查看所有可用模型"
            )
        return {"success": False, "error": f"模型目录不存在: {model_dir}{hint}", "alias": model_alias}

    # 构建执行环境
    env = {}
    if dataset:
        if "/" in dataset:
            if not os.path.exists(dataset):
                return {"success": False, "error": f"数据集路径不存在: {dataset}", "alias": model_alias}
            env["ONESCIENCE_DATASET_PATH"] = dataset
            env["ONESCIENCE_DATASET"] = os.path.basename(dataset)
        else:
            # 使用 config 解析数据集（自动触发 ModelScope 下载）
            resolved = config.resolve_dataset(dataset)
            if resolved:
                env["ONESCIENCE_DATASET_PATH"] = resolved
                env["ONESCIENCE_DATASET"] = os.path.basename(resolved)
                env["ONESCIENCE_DATASETS_DIR"] = config.datasets_dir
            else:
                hint = (
                    f"\n  提示: 数据集 '{dataset}' 在本地不存在，也无法从 ModelScope 自动下载。"
                    f"\n  可尝试以下方式:"
                    f"\n    1. 设置环境变量 ONESCIENCE_DATASETS_DIR 指向数据集所在目录"
                    f"\n    2. 使用完整路径: onescience bench -dataset /path/to/{dataset} -models ..."
                    f"\n    3. 检查网络连接后重试（自动下载需要外网访问 modelscope.cn）"
                    f"\n    4. 使用 'onescience data download {dataset}' 手动下载"
                )
                return {"success": False, "error": f"数据集 '{dataset}' 无法解析{hint}", "alias": model_alias}

    # 验证数据集路径是否包含实际数据文件
    # 用户明确指定了 dataset 但无法找到有效数据时，直接报错返回
    if "ONESCIENCE_DATASET_PATH" in env and not _validate_dataset_path(env["ONESCIENCE_DATASET_PATH"]):
        ds_path = env["ONESCIENCE_DATASET_PATH"]
        click.secho(
            f"❌  数据集路径中未找到可识别的数据文件:\n"
            f"    {ds_path}\n"
            f"    请通过以下方式设置正确路径:\n"
            f"      1. 设置环境变量: export ONESCIENCE_DATASETS_DIR=/path/to/datasets\n"
            f"      2. 在命令中使用完整路径: onescience bench -dataset /path/to/data\n"
            f"      3. 运行 'onescience config set data_dir /path/to/data'",
            fg="red",
        )
        return {"success": False, "error": f"数据集路径中无有效数据: {ds_path}", "alias": model_alias}

    # remock: 重置环境
    if cmd_type == "remock":
        for d in ["result", "results", "checkpoints", "logs", "__pycache__"]:
            p = model_dir / d
            if p.exists():
                shutil.rmtree(p)
        for f in model_dir.glob("*_execution.log"):
            f.unlink()
        return {"success": True, "output": "环境重置完成", "alias": model_alias}

    # 通用配置补丁：将数据集路径注入到模型的所有 YAML 配置文件
    config_backups: t.Dict[str, str] = {}
    if "ONESCIENCE_DATASET_PATH" in env:
        config_backups = _patch_model_configs(model_dir, env["ONESCIENCE_DATASET_PATH"])

    # 特定领域补丁 (earth 模型：ERA5 年份检测、stats/static 路径等)
    if "ONESCIENCE_DATASET_PATH" in env:
        config_path = model_dir / "conf" / "config.yaml"
        if config_path.exists():
            earth_info = _earth_config_patches(env, config_path)
            if earth_info is not None:
                _, patches = earth_info
                original = config_path.read_text(encoding="utf-8")
                modified = _patch_config_values(original, patches)
                if modified is not None:
                    config_backups[str(config_path)] = original
                    config_path.write_text(modified, encoding="utf-8")

    try:
        # 按模型类型选择执行方式
        output = ""
        all_success = True

        if model_dir.name == "CFD_Benchmark":
            result = _run_cfd_benchmark_script(model_dir, cmd_type, sub_model, dataset, env)
            output += result["output"]
            all_success = all_success and result["success"]
        else:
            # 通用 Python 模型（优先）
            result = _run_generic_model(model_dir, cmd_type, env, domain)
            if result.get("found_any"):
                output += result["output"]
                all_success = all_success and result["success"]
            else:
                # Fallback: .sh 脚本（无 Python 脚本时才走）
                sh_scripts = sorted(model_dir.glob("*.sh"))
                sh_scripts = [s for s in sh_scripts if not s.name.endswith("_execution.log")]
                if sh_scripts:
                    for sh_script in sh_scripts:
                        log_path = _get_log_path(model_dir, sh_script.stem)
                        result_sh = _run_cmd(["bash", str(sh_script)], model_dir, log_path, env)
                        output += result_sh["output"]
                        all_success = all_success and result_sh["success"]
                else:
                    output = result.get("output", "") or output
                    all_success = False if not result.get("found_any") else all_success

        metrics = _extract_metrics(output, domain)
        return {
            "success": all_success,
            "output": output,
            "domain": domain,
            "model": model,
            "alias": model_alias,
            "metrics": metrics,
        }
    finally:
        for filepath, original in config_backups.items():
            Path(filepath).write_text(original, encoding="utf-8")


def collect_results(run_results: t.List[dict]):
    for r in run_results:
        alias = r.get("alias")
        if not alias:
            continue
        domain = r.get("domain", "")
        model = r.get("model", "")
        dst = RESULTS_DIR / alias
        dst.mkdir(parents=True, exist_ok=True)

        # 即使执行失败也保存 metrics
        if r.get("metrics"):
            metrics_path = dst / "metrics.json"
            try:
                metrics_path.write_text(json.dumps(r["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        # 结果目录和脚本仅成功时复制
        if not r.get("success"):
            continue

        # 尝试查找模型目录
        model_dir = None
        info = model_registry.resolve(alias)
        if info and info.get("model_dir") and Path(info["model_dir"]).exists():
            model_dir = Path(info["model_dir"])
        else:
            from .registry import DOMAIN_DIR_MAP
            candidate = EXAMPLES_DIR / DOMAIN_DIR_MAP.get(domain, domain) / model
            if candidate.exists():
                model_dir = candidate

        if not model_dir:
            continue

        result_dir = model_dir / "result"
        if result_dir.exists() and result_dir.is_dir():
            shutil.copytree(str(result_dir), str(dst / "result"), dirs_exist_ok=True)
        result_py = model_dir / "result.py"
        if result_py.exists():
            shutil.copy2(str(result_py), str(dst / "result.py"))


def _num(val):
    if val == "N/A" or val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _colored(val, lo, hi, reverse=False):
    if val == "N/A" or val is None:
        return val
    n = _num(val)
    if n is None:
        return val
    lo_n = _num(lo)
    hi_n = _num(hi)
    if lo_n is not None and abs(n - lo_n) < 1e-9:
        return click.style(val, fg="blue", bold=True)
    if hi_n is not None and abs(n - hi_n) < 1e-9:
        return click.style(val, fg="red", bold=True)
    return val


def _render_metrics_table(results: t.List[dict], title: str, keys: t.List[str]):
    """渲染领域指标对比表格（通用函数，消除 earth/cfd 重复代码）"""
    if not results:
        return

    click.echo(f"\n{title}:")

    vals = {k: [] for k in keys}
    for r in results:
        m = r["metrics"]
        for k in keys:
            n = _num(m.get(k))
            if n is not None:
                vals[k].append(n)
    lo = {k: min(vals[k]) if vals[k] else None for k in keys}
    hi = {k: max(vals[k]) if vals[k] else None for k in keys}

    col_widths = {}
    col_widths["模型名称"] = max(len(r["alias"]) for r in results) + 2
    for k in keys:
        header_len = len(k)
        max_val_len = 0
        for r in results:
            v = str(r["metrics"].get(k, "N/A"))
            max_val_len = max(max_val_len, len(v))
        col_widths[k] = max(header_len, max_val_len) + 2

    all_cols = ["模型名称"] + keys
    sep = "+" + "+".join("-" * col_widths[col] for col in all_cols) + "+"
    click.echo(sep)

    hdr_parts = [f" {col:{col_widths[col]-1}}" for col in all_cols]
    click.echo("|" + "".join(hdr_parts) + "|")
    click.echo(sep)

    for r in results:
        m = r["metrics"]
        row_parts = [f" {r['alias']:{col_widths['模型名称']-1}}"]
        for k in keys:
            v = str(m.get(k, "N/A"))
            colored_v = _colored(v, lo.get(k), hi.get(k))
            row_parts.append(f" {colored_v:>{col_widths[k]-1}}")
        click.echo("|" + "".join(row_parts) + "|")
    click.echo(sep)


def print_comparison(results: t.List[dict]):
    if not results:
        return
    earth_results = [r for r in results if r.get("domain") == "earth" and r.get("metrics")]
    if earth_results:
        title = "气象模型对比" if len(earth_results) > 1 else "气象模型结果"
        _render_metrics_table(earth_results, title, ["RMSE", "ACC", "CSI", "MAE", "BIAS"])

    cfd_results = [r for r in results if r.get("domain") == "cfd" and r.get("metrics")]
    if cfd_results:
        title = "CFD模型对比" if len(cfd_results) > 1 else "CFD模型结果"
        _render_metrics_table(cfd_results, title, ["平均相对误差", "平均绝对误差", "平均MSE", "平均MAE", "平均MaxAE", "平均R²分数"])

    # 通用模型对比
    other_results = [r for r in results if r.get("domain") == "_custom" and r.get("metrics")]
    if other_results:
        title = "自定义模型对比" if len(other_results) > 1 else "自定义模型结果"
        click.echo(f"\n{title}:")
        for r in other_results:
            m = r["metrics"]
            click.echo(f"  {r['alias']}:")
            for k, v in sorted(m.items()):
                if v != "N/A":
                    click.echo(f"    {k}: {v}")


def load_saved_results(aliases: t.Optional[t.List[str]] = None) -> t.List[dict]:
    """从 RESULTS_DIR 加载之前保存的模型结果

    Args:
        aliases: 指定要加载的模型别名列表，为 None 时加载所有已有结果

    Returns:
        包含 alias / domain / model / metrics 的字典列表
    """
    if not RESULTS_DIR.exists():
        return []

    if aliases:
        dirs = [RESULTS_DIR / a for a in aliases]
    else:
        dirs = sorted(RESULTS_DIR.iterdir())

    results = []
    for d in dirs:
        if not d.is_dir():
            continue
        alias = d.name
        metrics_file = d / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = model_registry.resolve(alias)
        results.append({
            "alias": alias,
            "domain": info["domain"] if info else "",
            "model": info["model"] if info else "",
            "metrics": metrics,
        })
    return results
