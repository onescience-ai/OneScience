#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

ROOT = Path(__file__).resolve().parents[4]
BACKBONE_MOE = "onescience.models.UMA.uma_escn_moe"
BACKBONE_MD = "onescience.models.UMA.uma_escn_md"

RULES = [
    ("onescience.models.UMA.models.base", "onescience.models.UMA.base"),
    ("onescience.models.UMA.models.uma.escn_md", BACKBONE_MD),
    ("onescience.models.UMA.models.uma.escn_moe", BACKBONE_MOE),
    ("onescience.models.UMA.uma.escn_md", BACKBONE_MD),
    ("onescience.models.UMA.uma.escn_moe", BACKBONE_MOE),
    ("onescience.utils.uma.models.uma.escn_md", BACKBONE_MD),
    ("onescience.utils.uma.models.uma.escn_moe", BACKBONE_MOE),
    ("fairchem.core.models.base", "onescience.models.UMA.base"),
    ("fairchem.core.models.uma.escn_md", BACKBONE_MD),
    ("fairchem.core.models.uma.escn_moe", BACKBONE_MOE),
    ("onescience.models.UMA.units", "onescience.utils.uma.units"),
    ("onescience.models.UMA.modules", "onescience.utils.uma.modules"),
    ("onescience.models.UMA.common", "onescience.utils.uma.common"),
    ("fairchem.core.units", "onescience.utils.uma.units"),
    ("fairchem.core.modules", "onescience.utils.uma.modules"),
    ("fairchem.core.components", "onescience.utils.uma.components"),
    ("fairchem.core.common", "onescience.utils.uma.common"),
    ("fairchem.core", "onescience.utils.uma"),
]

LEGACY_PREFIX = (
    "onescience.models.UMA.models.",
    "onescience.models.UMA.uma.",
    "onescience.utils.uma.models.",
    "fairchem.core.models.",
)

def remap(s: str) -> str:
    prev = None
    cur = s
    while cur != prev:
        prev = cur
        for old, new in RULES:
            if cur == old or cur.startswith(old + "."):
                cur = new + cur[len(old):]
                break
    return cur

def to_py(x):
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=False)
    return x

def walk(x):
    if isinstance(x, str):
        return remap(x)
    if isinstance(x, dict):
        return {k: walk(v) for k, v in x.items()}
    if isinstance(x, list):
        return [walk(v) for v in x]
    return x

def find_legacy(x, path="root"):
    out = []
    if isinstance(x, str):
        if x.startswith(LEGACY_PREFIX):
            out.append((path, x))
    elif isinstance(x, dict):
        for k, v in x.items():
            out.extend(find_legacy(v, f"{path}.{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.extend(find_legacy(v, f"{path}[{i}]"))
    return out

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module_name, class_name):
        mapped = remap(module_name)
        try:
            return super().find_class(mapped, class_name)
        except ModuleNotFoundError:
            return super().find_class(module_name, class_name)

class CustomPickleModule:
    Unpickler = CustomUnpickler

def gf(obj, k):
    return obj[k] if isinstance(obj, dict) else getattr(obj, k)

def sf(obj, k, v):
    if isinstance(obj, dict):
        obj[k] = v
    else:
        setattr(obj, k, v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[*] loading: {args.src}")
    try:
        obj = torch.load(args.src, map_location="cpu", pickle_module=CustomPickleModule, weights_only=False)
    except TypeError:
        obj = torch.load(args.src, map_location="cpu", pickle_module=CustomPickleModule)

    mc = OmegaConf.create(walk(to_py(gf(obj, "model_config"))))
    tc = OmegaConf.create(walk(to_py(gf(obj, "tasks_config"))))
    sf(obj, "model_config", mc)
    sf(obj, "tasks_config", tc)

    cfg_py = OmegaConf.to_container(mc, resolve=False)
    bad = find_legacy(cfg_py)
    if bad:
        print("[FATAL] legacy paths still exist in model_config:")
        for p, v in bad[:20]:
            print(" ", p, "=", v)
        raise SystemExit(2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(out))
    print("[OK] saved:", out)
    print("[OK] model _target_:", mc.get("_target_", "<none>"))
    print("[OK] backbone model:", mc.get("backbone", {}).get("model"))

if __name__ == "__main__":
    main()
