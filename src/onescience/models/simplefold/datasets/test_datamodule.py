#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import pickle
import blobfile as bf
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from boltz_data_pipeline.types import Manifest
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.filter.dynamic.filter import DynamicFilter
from boltz_data_pipeline import const
from utils.datamodule_utils import load_input, collate, restype_3to1


class SimpleFoldPredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        num_repeats: int = 1,
    ):
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.num_repeats = num_repeats
        self.num_samples = len(self.manifest.records)
        print(f"num_samples: {self.num_samples}")

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, self.target_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        max_num_tokens = len(tokenized.tokens)

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
        sequence = ":".join(sequence)

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                max_atoms=None,
                max_tokens=None,
                symmetries={},
                compute_symmetries=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        features["aa_seq"] = sequence
        features["record"] = asdict(record)
        features["num_repeats"] = torch.tensor(self.num_repeats)
        features['max_num_tokens'] = torch.tensor(max_num_tokens, dtype=torch.long)
        features['cropped_num_tokens'] = torch.tensor(len(tokenized.tokens), dtype=torch.long)

        return features

    def __len__(self) -> int:
        return self.num_samples


class SimpleFoldInferenceDataModule(LightningDataModule):
    """DataModule for SimpleFold Inference."""

    def __init__(
        self,
        target_dir: str,
        num_workers: int,
        manifest_path: str = None,
        max_nsamples: int = None,
        filters: Optional[list[DynamicFilter]] = None,
        num_repeats: int = 1,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.target_dir = target_dir
        self.num_repeats = num_repeats

        self.target_dir = Path(self.target_dir)

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.target_dir / "manifest.json"
        manifest: Manifest = Manifest.load(self.manifest_path)
        records = manifest.records

        # Filter training records
        if filters is not None:
            records = [
                record
                for record in records
                if all(f.filter(record) for f in filters)
            ]

        if max_nsamples is not None:
            records = records[:max_nsamples]

        self.manifest = Manifest(records)

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = SimpleFoldPredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            num_repeats=self.num_repeats,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )
