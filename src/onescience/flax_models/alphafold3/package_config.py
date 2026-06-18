#!/usr/bin/env python3
"""
AlphaFold3包配置
提供给主setup.py的配置信息，隐藏所有实现细节
"""

import importlib.util
from pathlib import Path


ALPHAFOLD3_PACKAGE_DATA = {
    "onescience.flax_models.alphafold3": [
        "*.so",
        "*.dll",
        "*.dylib",
        "README.md",
        "test_data/**/*",
        "**/*.pyi",
    ],
    "onescience.flax_models.alphafold3.constants.converters": [
        "*.pickle",
    ],
}

ALPHAFOLD3_MANIFEST_RULES = [
    "include src/onescience/flax_models/alphafold3/*.so",
    "recursive-include src/onescience/flax_models/alphafold3/constants/converters *.pickle",
    "include src/onescience/flax_models/alphafold3/README.md",
    "recursive-include src/onescience/flax_models/alphafold3/test_data *",
    "global-exclude src/onescience/flax_models/alphafold3/*.cc",
    "global-exclude src/onescience/flax_models/alphafold3/*.cpp",
    "global-exclude src/onescience/flax_models/alphafold3/*.h",
    "global-exclude src/onescience/flax_models/alphafold3/*.hpp",
    "global-exclude src/onescience/flax_models/alphafold3/CMakeLists.txt",
]


def get_package_data():
    return ALPHAFOLD3_PACKAGE_DATA


def get_manifest_rules():
    return ALPHAFOLD3_MANIFEST_RULES


_build_done = False


def _load_af3_build_module():
    af3_build_path = Path(__file__).resolve().parents[2] / "_build" / "af3.py"
    spec = importlib.util.spec_from_file_location("onescience_af3_build", af3_build_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_hook(project_root, env, python_executable, subprocess_module):
    global _build_done
    if _build_done:
        return

    af3_build = _load_af3_build_module()

    print("[AF3] build hook triggered")
    try:
        print(f"[AF3] should_build={af3_build.should_build()} artifacts_exist={af3_build.artifacts_exist()} force_rebuild={af3_build.force_rebuild()}")
        af3_build.build_if_needed()
        print("[AF3] build hook finished")
    except af3_build.AF3BuildError as exc:
        print(f"[AF3] build skipped or failed: {exc}")
        if af3_build.is_strict():
            raise

    _build_done = True


def get_build_hook():
    return build_hook
