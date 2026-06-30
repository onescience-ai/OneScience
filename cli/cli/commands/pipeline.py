import click
from ..core.runner import run_model, collect_results, print_comparison
from ..core.registry import model_registry


def _parse_step(step_text: str, default_dataset: str):
    """解析单个步骤字符串

    格式: <cmd_type> <model_alias> [-dataset <dataset>]
    示例: train fno -dataset airfoil
          infer pangu -dataset era5
          eval fno
    """
    parts = step_text.strip().split()
    if len(parts) < 2:
        raise click.BadParameter(f"步骤格式错误，需要至少 2 个参数: {step_text}")

    cmd_type = parts[0]
    model_alias = parts[1]
    dataset = default_dataset

    i = 2
    while i < len(parts):
        if parts[i] == "-dataset" and i + 1 < len(parts):
            dataset = parts[i + 1]
            i += 2
        else:
            i += 1

    return cmd_type, model_alias, dataset


def _parse_steps(steps_text: str, default_dataset: str):
    """解析逗号分隔的步骤列表"""
    steps = []
    for step_text in steps_text.split(","):
        step_text = step_text.strip()
        if not step_text:
            continue
        cmd_type, model_alias, dataset = _parse_step(step_text, default_dataset)
        steps.append((cmd_type, model_alias, dataset))
    return steps


def _run_pipeline(steps, on_error):
    """依次执行 pipeline 步骤"""
    results = []

    for i, (cmd_type, model_alias, dataset) in enumerate(steps, 1):
        info = model_registry.resolve(model_alias)
        domain = info["domain"] if info else "?"
        desc = f"[{i}/{len(steps)}] {cmd_type} {model_alias} ({domain})"
        if dataset:
            desc += f" -dataset {dataset}"

        click.secho(f"\n{'=' * 48}", fg="bright_blue")
        click.secho(f"  {desc}", fg="bright_blue")
        click.secho(f"{'=' * 48}", fg="bright_blue")

        result = run_model(model_alias, cmd_type, dataset)
        results.append(result)

        if result["success"]:
            click.secho(f"  ✓ {model_alias} 执行成功", fg="green")
        else:
            err_msg = result.get("error") or result.get("output", "")
            click.secho(f"  ✗ {model_alias} 执行失败: {err_msg}", fg="red")

            if on_error == "stop":
                break
            elif on_error == "skip":
                click.secho(f"  跳过失败步骤，继续下一步", fg="yellow")

    return results


@click.command("pipeline")
@click.option("-steps", required=False, help="步骤列表，逗号分隔，格式: <cmd_type> <model> [-dataset <ds>], ...")
@click.option("-file", "pipeline_file", required=False, help="从文件读取 pipeline 定义")
@click.option("-dataset", "default_dataset", default="", help="默认数据集（步骤未指定时使用）")
@click.option("-on-error", "on_error", default="stop",
              type=click.Choice(["stop", "skip", "ignore"]),
              help="错误处理策略：stop=停止, skip=跳过, ignore=忽略")
@click.option("-compare", is_flag=True, help="执行完成后对比结果")
def pipeline(steps, pipeline_file, default_dataset, on_error, compare):
    """编排并执行多步骤工作流

    每个步骤独立指定模型、操作类型和数据集，按顺序依次执行。

    步骤格式: <cmd_type> <model_alias> [-dataset <dataset>]

    示例:
      onescience pipeline -steps "train fno -dataset airfoil, infer pangu -dataset era5"

      onescience pipeline -steps "train fno, infer pangu" -dataset era5

      onescience pipeline -steps "train fno -dataset airfoil, eval fno" -on-error skip -compare

      onescience pipeline -file workflow.txt -compare
    """
    # 从文件或参数获取步骤
    if pipeline_file:
        try:
            with open(pipeline_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                click.echo("pipeline 文件为空")
                return
            steps_text = content
        except FileNotFoundError:
            click.echo(f"文件不存在: {pipeline_file}")
            return
        except IOError as e:
            click.echo(f"读取文件失败: {e}")
            return
    elif steps:
        steps_text = steps
    else:
        click.echo("请指定 -steps 或 -file 参数")
        click.echo("示例: onescience pipeline -steps \"train fno -dataset airfoil, infer pangu\"")
        return

    # 解析步骤
    try:
        parsed_steps = _parse_steps(steps_text, default_dataset)
    except click.BadParameter as e:
        click.echo(f"步骤解析错误: {e}")
        return

    if not parsed_steps:
        click.echo("没有可执行的步骤")
        return

    click.secho(f"Pipeline 开始 - {len(parsed_steps)} 个步骤, 错误策略: {on_error}", fg="bright_green")

    # 依次执行
    results = _run_pipeline(parsed_steps, on_error)

    # 收集结果
    collect_results(results)
    click.echo(f"\nPipeline 执行完成，{len(results)} 个步骤")

    # 对比结果
    if compare and results:
        click.echo("\n结果对比:")
        print_comparison(results)
    elif compare:
        click.echo("没有结果可对比")
