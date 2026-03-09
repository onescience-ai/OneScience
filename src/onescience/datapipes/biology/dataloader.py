"""Dataloader factory for bioinformatics datasets.

Provides convenient dataloader creation functions for proteins, multimers,
and genomes.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from .datasets import ProteinDataset, MultimerDataset, GenomeDataset


def get_protein_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
    **kwargs
) -> DataLoader:
    """Create a dataloader for protein datasets.

    Args:
        data_path: Path to the dataset directory or file.
        batch_size: Number of samples per batch. Defaults to 1.
        shuffle: Whether to shuffle the data. Defaults to True.
        num_workers: Number of worker processes for data loading.
            Defaults to 0.
        distributed: Whether to use distributed sampling. Defaults to False.
        **kwargs: Additional arguments passed to ProteinDataset and DataLoader.

    Returns:
        DataLoader: Configured DataLoader for protein datasets.

    Raises:
        FileNotFoundError: If the data path does not exist.
        ValueError: If batch_size is not positive.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Create dataset
    dataset = ProteinDataset(
        data_path=data_path,
        **{k: v for k, v in kwargs.items() if k not in ['collate_fn', 'pin_memory', 'drop_last']}
    )

    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Disable shuffle when using sampler

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=kwargs.get('collate_fn', protein_collate_fn),
        pin_memory=kwargs.get('pin_memory', True),
        drop_last=kwargs.get('drop_last', False),
    )

    return dataloader


def get_multimer_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
    **kwargs
) -> DataLoader:
    """Create a dataloader for multimer datasets.

    Args:
        data_path: Path to the dataset directory or file.
        batch_size: Number of samples per batch. Defaults to 1.
        shuffle: Whether to shuffle the data. Defaults to True.
        num_workers: Number of worker processes for data loading.
            Defaults to 0.
        distributed: Whether to use distributed sampling. Defaults to False.
        **kwargs: Additional arguments passed to MultimerDataset and DataLoader.

    Returns:
        DataLoader: Configured DataLoader for multimer datasets.

    Raises:
        FileNotFoundError: If the data path does not exist.
        ValueError: If batch_size is not positive.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Create dataset
    dataset = MultimerDataset(
        data_path=data_path,
        **{k: v for k, v in kwargs.items() if k not in ['collate_fn', 'pin_memory', 'drop_last']}
    )

    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=kwargs.get('collate_fn', multimer_collate_fn),
        pin_memory=kwargs.get('pin_memory', True),
        drop_last=kwargs.get('drop_last', False),
    )

    return dataloader


def get_genome_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
    **kwargs
) -> DataLoader:
    """Create a dataloader for genome datasets.

    Args:
        data_path: Path to the dataset directory or file.
        batch_size: Number of samples per batch. Defaults to 1.
        shuffle: Whether to shuffle the data. Defaults to True.
        num_workers: Number of worker processes for data loading.
            Defaults to 0.
        distributed: Whether to use distributed sampling. Defaults to False.
        **kwargs: Additional arguments passed to GenomeDataset and DataLoader.

    Returns:
        DataLoader: Configured DataLoader for genome datasets.

    Raises:
        FileNotFoundError: If the data path does not exist.
        ValueError: If batch_size is not positive.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Create dataset
    dataset = GenomeDataset(
        data_path=data_path,
        **{k: v for k, v in kwargs.items() if k not in ['collate_fn', 'pin_memory', 'drop_last']}
    )

    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=kwargs.get('collate_fn', genome_collate_fn),
        pin_memory=kwargs.get('pin_memory', True),
        drop_last=kwargs.get('drop_last', False),
    )

    return dataloader


def protein_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for protein batches.

    Handles variable-length sequences by padding.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched data dictionary with padded sequences.
    """
    if not batch:
        return {}

    # Get keys from first sample
    keys = batch[0].keys()
    result = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # Handle different data types
        if key in ['sequence', 'seq']:
            # Pad sequences
            max_len = max(len(v) for v in values)
            padded = []
            masks = []
            for v in values:
                pad_len = max_len - len(v)
                if isinstance(v, str):
                    padded.append(v + '-' * pad_len)
                    masks.append([1] * len(v) + [0] * pad_len)
                else:
                    padded.append(np.pad(v, (0, pad_len), mode='constant'))
                    masks.append([1] * len(v) + [0] * pad_len)
            result[key] = padded
            result[f'{key}_mask'] = torch.tensor(masks, dtype=torch.bool)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.tensor(np.stack(values))
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values

    return result


def multimer_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for multimer batches.

    Handles multiple chains with variable lengths.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched data dictionary with padded sequences and chain information.
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    result = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if key in ['sequences', 'chains']:
            # Handle multiple chains
            max_chains = max(len(v) for v in values)
            max_len = max(
                max(len(seq) for seq in chains) if chains else 0
                for chains in values
            )

            padded_chains = []
            chain_masks = []
            for chains in values:
                padded_seqs = []
                masks = []
                for seq in chains:
                    pad_len = max_len - len(seq)
                    if isinstance(seq, str):
                        padded_seqs.append(seq + '-' * pad_len)
                    else:
                        padded_seqs.append(np.pad(seq, (0, pad_len), mode='constant'))
                    masks.append([1] * len(seq) + [0] * pad_len)

                # Pad number of chains
                while len(padded_seqs) < max_chains:
                    if isinstance(chains[0], str) if chains else True:
                        padded_seqs.append('-' * max_len)
                    else:
                        padded_seqs.append(np.zeros(max_len, dtype=chains[0].dtype if chains else np.float32))
                    masks.append([0] * max_len)

                padded_chains.append(padded_seqs)
                chain_masks.append(masks)

            result[key] = padded_chains
            result[f'{key}_mask'] = torch.tensor(chain_masks, dtype=torch.bool)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.tensor(np.stack(values))
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values

    return result


def genome_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for genome batches.

    Handles long sequences with potential windowing.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched data dictionary.
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    result = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if key in ['sequence', 'seq', 'genome']:
            # Genome sequences may be very long, use windowing or truncation
            max_len = max(len(v) for v in values)
            # Limit max length to prevent OOM
            max_len = min(max_len, 100000)

            padded = []
            masks = []
            for v in values:
                v = v[:max_len]  # Truncate if too long
                pad_len = max_len - len(v)
                if isinstance(v, str):
                    padded.append(v + 'N' * pad_len)
                    masks.append([1] * len(v) + [0] * pad_len)
                else:
                    padded.append(np.pad(v, (0, pad_len), mode='constant'))
                    masks.append([1] * len(v) + [0] * pad_len)

            result[key] = padded
            result[f'{key}_mask'] = torch.tensor(masks, dtype=torch.bool)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.tensor(np.stack(values))
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values

    return result
