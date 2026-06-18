#!/usr/bin/env python3
"""Validate a lightweight OneScience bio inference manifest without PyYAML."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_MANIFEST_REQUIRED_KEYS = {
    "model_family",
    "inference_mode",
    "cwd",
    "entrypoint",
    "command",
    "inputs",
    "checkpoint",
    "outputs",
}

REQUEST_REQUIRED_KEYS = {
    "model_family",
    "inference_mode",
    "runtime",
}

KNOWN_MODELS = {
    "AlphaFold",
    "OpenFold",
    "AlphaFold3",
    "Protenix",
    "SimpleFold",
    "RFdiffusion",
    "ProteinMPNN",
    "PT-DiT",
    "ProToken",
    "Evo2",
    "MolSculptor",
}


def top_level_yaml_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        if line.startswith("#") or not line.strip() or line.startswith((" ", "\t", "-")):
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:", line)
        if match:
            keys.add(match.group(1))
    return keys


def load_manifest(path: Path) -> tuple[set[str], str | None, str]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON manifest root must be an object")
        keys = set(data.keys())
        return keys, data.get("model_family"), text
    keys = top_level_yaml_keys(text)
    model_match = re.search(r"^model_family\s*:\s*['\"]?([^'\"\n#]+)", text, re.MULTILINE)
    model = model_match.group(1).strip() if model_match else None
    return keys, model, text


def required_keys_for(keys: set[str]) -> set[str]:
    if "runtime" in keys:
        return REQUEST_REQUIRED_KEYS
    return RUN_MANIFEST_REQUIRED_KEYS


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate required fields in a bio inference manifest.")
    parser.add_argument("manifest")
    args = parser.parse_args()

    path = Path(args.manifest)
    keys, model, text = load_manifest(path)
    missing = sorted(required_keys_for(keys) - keys)
    warnings = []
    if model and model not in KNOWN_MODELS:
        warnings.append(f"unknown model_family: {model}")
    if "runtime" in keys and not re.search(r"^\s{2}cwd\s*:", text, re.MULTILINE):
        warnings.append("runtime.cwd is not declared")
    if "runtime" in keys and not re.search(r"^\s{2}entrypoint\s*:", text, re.MULTILINE):
        warnings.append("runtime.entrypoint is not declared")
    result = {
        "manifest": str(path),
        "missing_required_keys": missing,
        "model_family": model,
        "warnings": warnings,
        "valid": not missing,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
