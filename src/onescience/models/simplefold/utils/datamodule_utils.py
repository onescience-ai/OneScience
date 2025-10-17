#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from torch import Tensor
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from boltz_data_pipeline.feature.pad import pad_to_max
from boltz_data_pipeline.crop.cropper import Cropper
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from processor.protein_processor import ProteinDataProcessor
from boltz_data_pipeline.types import Connection, Input, Manifest, Record, Structure
from boltz_data_pipeline import const


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
restype_3to1["UNK"] = "X"
restype_3to1["-"] = "X"
restype_3to1["<pad>"] = "X"


@dataclass
class Dataset:
    """Data holder."""
    tokenized_dir: str
    target_dir: Path
    esm_dir: str
    connectivity_dir: str
    cheap_dir: str
    manifest: Manifest
    cropper: Cropper
    tokenizer: BoltzTokenizer
    featurizer: BoltzFeaturizer
    cluster: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_name: str
    tokenized_dir: str
    target_dir: Path
    cropper: Cropper
    filters: Optional[list] = None
    # split: Optional[str] = None
    manifest_path: Optional[str] = None
    esm_dir: Optional[str] = None
    connectivity_dir: Optional[str] = None
    cheap_dir: Optional[str] = None
    record_list: Optional[str] = None
    cluster: Optional[str] = None


def load_input(record: Record, target_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
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


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "aa_seq",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


def extract_sequence_from_tokens(tokenized):
    seq = []
    sequence = []
    current_entity = 0
    for i, t in enumerate(tokenized.tokens):
        entity = t[7]
        if entity != current_entity:
            sequence.append("".join(seq))
            seq = []
            current_entity = entity

        res_type = t[4]
        res_name = restype_3to1[const.tokens[res_type]]
        seq.append(res_name)
        if i == len(tokenized.tokens) - 1:
            sequence.append("".join(seq))
            seq = []
            current_entity = entity
    return ":".join(sequence)


def process_one_inference_structure(
    structure_path, 
    record_path,
    tokenizer: BoltzTokenizer,
    featurizer: BoltzFeaturizer,
    processor: ProteinDataProcessor,
    esm_model=None,
    esm_dict=None,
    af2_to_esm=None,
):
    structure: Structure = Structure.load(structure_path)
    input_data = Input(structure, {})
    record = json.load(open(record_path))

    tokenized = tokenizer.tokenize(input_data)
    sequence = extract_sequence_from_tokens(tokenized)
    features = featurizer.process(tokenized)

    features["aa_seq"] = sequence
    features["record"] = record
    features["num_repeats"] = torch.tensor(1)
    features['max_num_tokens'] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
    features['cropped_num_tokens'] = torch.tensor(len(tokenized.tokens), dtype=torch.long)

    batch = collate([features])

    batch = processor.preprocess_inference(
        batch,
        esm_model=esm_model,
        esm_dict=esm_dict,
        af2_to_esm=af2_to_esm,
    )

    return batch, structure, Record(**record)
