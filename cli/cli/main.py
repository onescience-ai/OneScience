import os
import click


def _detect_device() -> str:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return "gpu"
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        pass
    try:
        r = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return "dcu"
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        pass
    return "cpu"


def _auto_bootstrap():
    if _auto_bootstrap._done:
        return
    _auto_bootstrap._done = True

    if "device" not in os.environ:
        os.environ["device"] = _detect_device()

    if "ONESCIENCE_DATASETS_DIR" not in os.environ:
        from .road import ONESCIENCE_DATASETS_DIR
        from .core.config import _resolve_writable_dir
        os.environ["ONESCIENCE_DATASETS_DIR"] = _resolve_writable_dir(
            "datasets_dir", ONESCIENCE_DATASETS_DIR, "datasets"
        )
    if "ONESCIENCE_MODELS_DIR" not in os.environ:
        from .road import ONESCIENCE_MODELS_DIR
        from .core.config import _resolve_writable_dir
        os.environ["ONESCIENCE_MODELS_DIR"] = _resolve_writable_dir(
            "models_dir", ONESCIENCE_MODELS_DIR, "models"
        )


_auto_bootstrap._done = False


class CLI(click.Group):
    def main(self, *args, **kwargs):
        try:
            return super().main(*args, **kwargs)
        except SystemExit as e:
            from .status_codes import ExitStatus
            raise SystemExit(e.code if e.code else ExitStatus.ERROR)


@click.group(cls=CLI)
@click.version_option(version="0.3.0")
def cli():
    """OneScience 科学计算 CLI 工具"""
    _auto_bootstrap()


from .commands import commands_dict

for name, cmd in commands_dict.items():
    cli.add_command(cmd, name=name)

if __name__ == "__main__":
    cli()
