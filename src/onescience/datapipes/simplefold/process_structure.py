#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz, 
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. 

import os
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import asdict, replace, dataclass

from onescience.datapipes.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from onescience.datapipes.boltz_data_pipeline.types import Connection, Input, Manifest, Record, Structure


@dataclass
class Sample:
    record: Record
    chain_id: Optional[int] = None
    interface_id: Optional[int] = None


def load_input(record: Record, target_dir: Path) -> Input:
    # Load processed structure
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )
    return Input(structure, {})


def tokenize_structure(
    record,
    tokenizer,
    target_dir: Path,
    save_token_dir: str,
    save_token_record_dir: Path,
):
    chains = [c for c in record.chains if c.valid]
    interfaces = [i for i in record.interfaces if i.valid]
    record = replace(record, chains=chains, interfaces=interfaces)
    sample = Sample(record=record)

    try:
        input_data = load_input(sample.record, target_dir)
    except Exception as e:
        print(f"Failed to load input for {sample.record.id} with error {e}. Skipping.")
        return False

    try:
        tokenized = tokenizer.tokenize(input_data)
    except Exception as e:
        print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
        return False

    # save tokenized data as pickle
    with open(os.path.join(save_token_dir, f"{sample.record.id}.pkl"), "wb") as f:
        pickle.dump(tokenized, f)

    token_record_path = save_token_record_dir / f"{sample.record.id}.json"
    with token_record_path.open("w") as f:
        json.dump(asdict(sample.record), f)

    return True


def finalize(outdir: Path) -> None:
    # Group records into a manifest
    records_dir = outdir / "records"
    failed_count = 0
    records = []
    for record in records_dir.iterdir():
        try:
            with record.open("r") as f:
                records.append(json.load(f))
        except:
            failed_count += 1
            print(f"Failed to parse {record}")
    print(f"Failed to parse {failed_count} entries")

    # Save manifest
    outpath = outdir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)


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
        success = tokenize_structure(
            record,
            tokenizer,
            target_dir,
            str(save_token_dir),
            save_token_record_dir,
        )

    finalize(Path(args.token_dir))