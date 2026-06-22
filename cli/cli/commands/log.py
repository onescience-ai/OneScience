import os
import re
import time
import click
from pathlib import Path
from datetime import datetime
from ..core.registry import model_registry, EXAMPLES_DIR
from ..core.runner import RESULTS_DIR


@click.command("log")
@click.argument("model_aliases", required=False)
@click.option("-tail", default=50, type=int, help="显示日志末尾行数")
@click.option("-head", default=0, type=int, help="显示日志开头行数")
@click.option("-search", default=None, help="搜索关键词")
@click.option("-all", "show_all", is_flag=True, help="显示完整日志")
@click.option("-type", "log_type", default="all", type=click.Choice(["all", "train", "infer", "eval"]))
@click.option("-follow", is_flag=True, help="实时跟踪日志")
@click.option("-date", "log_date", default=None, help="查看指定日期日志 (YYYY-MM-DD)")
@click.option("-clean", default=0, type=int, help="清理指定天数前的旧日志")
def log(model_aliases, tail, head, search, show_all, log_type, follow, log_date, clean):
    """查看模型日志"""
    if not model_aliases:
        click.echo("错误: 缺少模型别名")
        return
    if clean > 0:
        _clean_all(model_aliases, clean)
        return
    aliases = [a.strip() for a in model_aliases.split(",") if a.strip()]
    for alias in aliases:
        info = model_registry.resolve(alias)
        if not info:
            click.secho(f"未知模型: {alias}", fg="red")
            continue
        model_dir = info.get("model_dir")
        if not model_dir:
            from ..core.registry import DOMAIN_DIR_MAP
            model_dir = EXAMPLES_DIR / DOMAIN_DIR_MAP.get(info["domain"], info["domain"]) / info["model"]
        model_dir = Path(model_dir)
        log_files = _find_log_files(model_dir, log_type, info.get("sub_model", ""), alias)
        if log_date:
            log_files = _filter_by_date(log_files, log_date)
        if not log_files:
            click.echo(f"未找到 {alias} 的日志文件")
            continue
        click.secho(f"模型: {alias}", fg="green")
        for lf in log_files[:5]:
            click.echo(f"--- {lf.name} ---")
            if show_all:
                click.echo(lf.read_text(encoding="utf-8", errors="ignore"))
            elif follow:
                _tail_follow(lf, tail, alias)
            elif head > 0:
                lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()[:head]
                click.echo("\n".join(lines))
            elif search:
                for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if search in line:
                        click.echo(line)
            else:
                lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
                click.echo("\n".join(lines[-tail:]))


def _find_log_files(model_dir: Path, log_type: str, sub_model: str, alias: str) -> list:
    files = []
    patterns = []
    if log_type == "train" or log_type == "all":
        patterns.extend(["train_execution.log", f"{alias}_execution.log", "*.log"])
    if log_type == "infer" or log_type == "all":
        patterns.append("inference_execution.log")
    if log_type == "eval" or log_type == "all":
        patterns.append("result_execution.log")
    for p in patterns:
        for f in model_dir.glob(p):
            if f.is_file():
                files.append(f)
    for subdir in ["logs", "lightning_logs"]:
        d = model_dir / subdir
        if d.exists():
            files.extend(d.glob("*.log"))
            files.extend(d.glob("*.log.*"))
    result_dir = RESULTS_DIR / alias
    if result_dir.exists():
        files.extend(result_dir.glob("*.log"))
        files.extend(result_dir.glob("*.log.*"))
    return list(set(files))


def _filter_by_date(log_files: list, log_date: str) -> list:
    filtered = []
    for f in log_files:
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime.strftime("%Y-%m-%d") == log_date:
                filtered.append(f)
        except OSError:
            pass
    return filtered


def _clean_all(model_aliases: str, days: int):
    click.echo(f"正在清理 {days} 天前的旧日志...")
    aliases = [a.strip() for a in model_aliases.split(",") if a.strip()]
    for alias in aliases:
        info = model_registry.resolve(alias)
        if not info:
            continue
        model_dir = info.get("model_dir")
        if not model_dir:
            from ..core.registry import DOMAIN_DIR_MAP
            model_dir = EXAMPLES_DIR / DOMAIN_DIR_MAP.get(info["domain"], info["domain"]) / info["model"]
        model_dir = Path(model_dir)
        log_files = _find_log_files(model_dir, "all", info.get("sub_model", ""), alias)
        _clean_old_logs(log_files, days, alias)
    click.echo("日志清理完成")


def _clean_old_logs(log_files: list, days: int, alias: str):
    now = datetime.now().timestamp()
    count = 0
    for f in log_files:
        mtime = f.stat().st_mtime
        if (now - mtime) > days * 86400:
            f.unlink()
            count += 1
    if count > 0:
        click.echo(f"已清理 {alias}: {count} 个日志文件")


def _tail_follow(log_path: Path, n: int, alias: str):
    click.echo(f"实时跟踪 {alias} 日志 (Ctrl+C 退出)...")
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines[-n:]:
        click.echo(line)
    last_size = log_path.stat().st_size
    try:
        while True:
            time.sleep(0.5)
            new_size = log_path.stat().st_size
            if new_size > last_size:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_size)
                    for line in f:
                        click.echo(line.rstrip())
                last_size = new_size
    except KeyboardInterrupt:
        pass
