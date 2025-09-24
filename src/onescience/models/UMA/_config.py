

from __future__ import annotations

import os
import shutil

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache/fairchem")
os.makedirs(CACHE_DIR, exist_ok=True)


def clear_cache():
    try:
        shutil.rmtree(CACHE_DIR)
    except FileNotFoundError:
        print(f"No FAIRChem cache directory found at {CACHE_DIR}")
