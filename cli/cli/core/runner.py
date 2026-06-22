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
from .registry import model_registry, EXAMPLES_DIR, PROJECT_ROOT, SRC_DIR
from .config import config
from ..road import DATASET_PATHS


def _patch_config_values(content: str, patches: t.Dict[str, str]) -> t.Optional[str]:
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


def _earth_config_patches(env: dict) -> t.Optional[t.Tuple[str, dict]]:
    data_dir = env.get("ONESCIENCE_DATASET_PATH", "")
    if not data_dir:
        return None
    patches = {}
    patches["datapipe.dataset.data_dir"] = f"'{data_dir}'"
    patches["datapipe.dataset.stats_dir"] = f"'{data_dir}/stats/'"
    patches["datapipe.dataset.static_dir"] = f"'{data_dir}/static/'"

    data_glob = os.path.join(data_dir, "data", "*.h5")
    h5_files = sorted(glob.glob(data_glob))
    if h5_files:
        years = sorted(int(os.path.basename(f).replace(".h5", "")) for f in h5_files)
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

    return (data_dir, patches)


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


_CMD_TIMEOUT = 7200


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


def _run_cmd(cmd: t.List[str], cwd: Path, log_path: Path, env: t.Optional[dict] = None) -> dict:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # 仅当 onescience 源码目录存在时才注入 PYTHONPATH
    # 确保自定义项目不受 PYTHONPATH 污染
    if SRC_DIR.exists():
        src_parent = str(SRC_DIR.parent)
        existing_pypath = merged_env.get("PYTHONPATH", "")
        pypath_parts = [p for p in existing_pypath.split(os.pathsep) if p and p != src_parent]
        pypath_parts.insert(0, src_parent)
        merged_env["PYTHONPATH"] = os.pathsep.join(pypath_parts)

    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, env=merged_env,
            timeout=_CMD_TIMEOUT,
        )
        output = result.stdout + result.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return {"success": result.returncode == 0, "output": output, "log_path": log_path}
    except subprocess.TimeoutExpired:
        msg = f"命令执行超时（{_CMD_TIMEOUT}秒）: {' '.join(cmd)}"
        log_path.write_text(msg, encoding="utf-8")
        return {"success": False, "output": msg, "log_path": log_path}
    except Exception as e:
        log_path.write_text(str(e), encoding="utf-8")
        return {"success": False, "output": str(e), "log_path": log_path}


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


def _run_generic_model(model_dir: Path, cmd_type: str, env: dict, domain: str) -> dict:
    """通用模型执行器

    扫描模型目录下的脚本，按命令类型选择执行：
      - train: train.py
      - infer: inference.py / infer.py
      - eval:  result.py

    如果标准脚本不存在，fallback 扫描目录下所有 .py 文件。
    """
    output = ""
    all_success = True

    script_map = {
        "TRAIN": ["train.py"],
        "INFER": ["inference.py", "infer.py"],
        "EVAL": ["result.py"],
        "train": ["train.py", "inference.py", "infer.py", "result.py"],
        "infer": ["inference.py", "infer.py"],
        "eval": ["result.py"],
    }
    scripts = script_map.get(cmd_type, ["train.py", "inference.py", "result.py"])

    found = False
    for script in scripts:
        script_path = model_dir / script
        if script_path.exists():
            found = True
            log_prefix = script.replace(".py", "")
            log_path = _get_log_path(model_dir, log_prefix)
            result = _run_cmd(["python", script], model_dir, log_path, env)
            output += result["output"]
            all_success = all_success and result["success"]
            if not result["success"] and cmd_type == "train":
                break

    # Fallback: 标准脚本都不存在时，扫描目录下所有非 test 的 .py 文件
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
        return {"success": False, "error": f"未知模型: {model_alias}"}

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
        return {"success": False, "error": f"模型目录不存在: {model_dir}"}

    # 构建执行环境
    env = {}
    if dataset:
        if "/" in dataset:
            env["ONESCIENCE_DATASET_PATH"] = dataset
            env["ONESCIENCE_DATASET"] = os.path.basename(dataset)
        else:
            datasets_dir = os.environ.get("ONESCIENCE_DATASETS_DIR", "")
            if dataset in DATASET_PATHS:
                ds_path = str(Path(datasets_dir) / DATASET_PATHS[dataset])
                env["ONESCIENCE_DATASET_PATH"] = ds_path
                env["ONESCIENCE_DATASETS_DIR"] = datasets_dir
            elif datasets_dir and (Path(datasets_dir) / dataset).exists():
                ds_path = str(Path(datasets_dir) / dataset)
                env["ONESCIENCE_DATASET_PATH"] = ds_path
                env["ONESCIENCE_DATASETS_DIR"] = datasets_dir
            env["ONESCIENCE_DATASET"] = dataset

    # remock: 重置环境
    if cmd_type == "remock":
        for d in ["result", "results", "checkpoints", "logs", "__pycache__"]:
            p = model_dir / d
            if p.exists():
                shutil.rmtree(p)
        for f in model_dir.glob("*_execution.log"):
            f.unlink()
        return {"success": True, "output": "环境重置完成"}

    # 配置打补丁 (earth 模型特有)
    config_backup = None
    config_path = model_dir / "conf" / "config.yaml"
    if "ONESCIENCE_DATASET_PATH" in env and config_path.exists():
        earth_info = _earth_config_patches(env)
        if earth_info is not None:
            _, patches = earth_info
            original = config_path.read_text(encoding="utf-8")
            modified = _patch_config_values(original, patches)
            if modified is not None:
                config_backup = original
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
        if config_backup is not None:
            config_path.write_text(config_backup, encoding="utf-8")


def collect_results(run_results: t.List[dict]):
    for r in run_results:
        if not r.get("success"):
            continue
        alias = r["alias"]
        domain = r["domain"]
        model = r["model"]
        dst = RESULTS_DIR / alias
        dst.mkdir(parents=True, exist_ok=True)

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
        if r.get("metrics"):
            metrics_path = dst / "metrics.json"
            try:
                metrics_path.write_text(json.dumps(r["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass


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


def print_comparison(results: t.List[dict]):
    if not results:
        return
    earth_results = [r for r in results if r.get("domain") == "earth" and r.get("metrics")]
    if earth_results:
        title = "气象模型对比" if len(earth_results) > 1 else "气象模型结果"
        click.echo(f"\n{title}:")
        earth_keys = ["RMSE", "ACC", "CSI", "MAE", "BIAS"]

        vals = {k: [] for k in earth_keys}
        for r in earth_results:
            m = r["metrics"]
            for k in earth_keys:
                n = _num(m.get(k))
                if n is not None:
                    vals[k].append(n)
        lo = {k: min(vals[k]) if vals[k] else None for k in earth_keys}
        hi = {k: max(vals[k]) if vals[k] else None for k in earth_keys}

        col_widths = {}
        col_widths["模型名称"] = max(len(r["alias"]) for r in earth_results) + 2
        for k in earth_keys:
            header_len = len(k)
            max_val_len = 0
            for r in earth_results:
                v = str(r["metrics"].get(k, "N/A"))
                max_val_len = max(max_val_len, len(v))
            col_widths[k] = max(header_len, max_val_len) + 2

        sep = "+" + "+".join("-" * col_widths[col] for col in ["模型名称"] + earth_keys) + "+"
        click.echo(sep)

        hdr_parts = []
        for col in ["模型名称"] + earth_keys:
            hdr_parts.append(f" {col:{col_widths[col]-1}}")
        hdr = "|" + "".join(hdr_parts) + "|"
        click.echo(hdr)
        click.echo(sep)

        for r in earth_results:
            m = r["metrics"]
            row_parts = [f" {r['alias']:{col_widths['模型名称']-1}}"]
            for k in earth_keys:
                v = str(m.get(k, "N/A"))
                colored_v = _colored(v, lo.get(k), hi.get(k))
                row_parts.append(f" {colored_v:>{col_widths[k]-1}}")
            click.echo("|" + "".join(row_parts) + "|")
        click.echo(sep)

    cfd_results = [r for r in results if r.get("domain") == "cfd" and r.get("metrics")]
    if cfd_results:
        title = "CFD模型对比" if len(cfd_results) > 1 else "CFD模型结果"
        click.echo(f"\n{title}:")
        cfd_keys = ["平均相对误差", "平均绝对误差", "平均MSE", "平均MAE", "平均MaxAE", "平均R²分数"]

        vals = {k: [] for k in cfd_keys}
        for r in cfd_results:
            m = r["metrics"]
            for k in cfd_keys:
                n = _num(m.get(k))
                if n is not None:
                    vals[k].append(n)
        lo = {k: min(vals[k]) if vals[k] else None for k in cfd_keys}
        hi = {k: max(vals[k]) if vals[k] else None for k in cfd_keys}

        col_widths = {}
        col_widths["模型名称"] = max(len(r["alias"]) for r in cfd_results) + 2
        for k in cfd_keys:
            header_len = len(k)
            max_val_len = 0
            for r in cfd_results:
                v = str(r["metrics"].get(k, "N/A"))
                max_val_len = max(max_val_len, len(v))
            col_widths[k] = max(header_len, max_val_len) + 2

        sep = "+" + "+".join("-" * col_widths[col] for col in ["模型名称"] + cfd_keys) + "+"
        click.echo(sep)

        hdr_parts = []
        for col in ["模型名称"] + cfd_keys:
            hdr_parts.append(f" {col:{col_widths[col]-1}}")
        hdr = "|" + "".join(hdr_parts) + "|"
        click.echo(hdr)
        click.echo(sep)

        for r in cfd_results:
            m = r["metrics"]
            row_parts = [f" {r['alias']:{col_widths['模型名称']-1}}"]
            for k in cfd_keys:
                v = str(m.get(k, "N/A"))
                colored_v = _colored(v, lo.get(k), hi.get(k))
                row_parts.append(f" {colored_v:>{col_widths[k]-1}}")
            click.echo("|" + "".join(row_parts) + "|")
        click.echo(sep)

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
