#!/usr/bin/env python3
"""Check whether a bio inference output directory contains expected artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_SUFFIXES = {
    "AlphaFold": [".pdb", ".json", ".pkl"],
    "OpenFold": [".pdb", ".pkl"],
    "AlphaFold3": [".cif", ".pdb", ".json", ".npz"],
    "Protenix": [".cif", ".pdb", ".json", ".npz"],
    "SimpleFold": [".pdb", ".cif"],
    "RFdiffusion": [".pdb", ".trb"],
    "ProteinMPNN": [".fa", ".fasta", ".jsonl", ".npz"],
    "PT-DiT": [".pkl", ".pdb"],
    "ProToken": [".pkl", ".pdb", ".txt"],
    "Evo2": [".json", ".csv", ".txt", ".pt", ".npy"],
    "MolSculptor": [".smi", ".csv", ".pkl", ".sdf"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect model inference output artifacts.")
    parser.add_argument("model_family", choices=sorted(EXPECTED_SUFFIXES))
    parser.add_argument("output_dir")
    parser.add_argument("--min-files", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.output_dir)
    suffixes = EXPECTED_SUFFIXES[args.model_family]
    matches = [path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes]
    result = {
        "model_family": args.model_family,
        "output_dir": str(root),
        "exists": root.exists(),
        "expected_suffixes": suffixes,
        "matched_count": len(matches),
        "matched_files": [str(path) for path in matches[:50]],
        "valid": root.exists() and len(matches) >= args.min_files,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
