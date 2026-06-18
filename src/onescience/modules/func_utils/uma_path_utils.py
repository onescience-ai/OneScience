"""Path helpers for UMA runtime assets."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_jd_path(jd_path: str | None = None) -> str:
    """Resolve path to ``Jd.pt`` with explicit/configurable precedence.

    Args:
        jd_path (str | None): Optional explicit path from model config.

    Returns:
        str: Absolute path to the resolved ``Jd.pt`` file.

    Raises:
        FileNotFoundError: If no candidate path contains ``Jd.pt``.
    """
    candidates: list[Path] = []
    if jd_path:
        candidates.append(Path(jd_path))

    env_path = os.environ.get("ONESCIENCE_UMA_JD_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path.cwd() / "models" / "Jd.pt")
    candidates.append(Path(__file__).resolve().parents[3] / "models" / "Jd.pt")

    for path in candidates:
        if path.is_file():
            return str(path)

    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Unable to locate UMA rotation basis file 'Jd.pt'. "
        f"Checked: {checked}. "
        "Set model config `jd_path` or env `ONESCIENCE_UMA_JD_PATH`."
    )
