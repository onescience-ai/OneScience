#!/usr/bin/env python3
"""Compatibility entry point for AlphaFold3 build helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from onescience._build.af3 import AF3BuildError, build_all


if __name__ == "__main__":
    try:
        build_all()
    except AF3BuildError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    sys.exit(0)
