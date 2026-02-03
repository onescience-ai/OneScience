#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import argparse
from pathlib import Path
from tqdm import tqdm

from onescience.datapipes.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from onescience.datapipes.boltz_data_pipeline.types import Manifest
from onescience.datapipes.simplefold.process_structure import tokenize_structure, finalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize structure data.")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory containing the processed structure data.",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        required=True,
        help="Directory to save the tokenized data.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    manifest_path = target_dir / "manifest.json"
    manifest: Manifest = Manifest.load(manifest_path)
    tokenizer = BoltzTokenizer()
    records = manifest.records
    print(f"Number of records after filtering: {len(records)}")

    save_token_dir = Path(args.token_dir) / "tokens"
    save_token_record_dir = Path(args.token_dir) / "records"
    save_token_dir.mkdir(parents=True, exist_ok=True)
    save_token_record_dir.mkdir(parents=True, exist_ok=True)

    for record in tqdm(records):
        tokenize_structure(
            record,
            tokenizer,
            target_dir,
            str(save_token_dir),
            save_token_record_dir,
        )

    finalize(Path(args.token_dir))
