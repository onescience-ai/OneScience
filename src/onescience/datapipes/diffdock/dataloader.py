from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        if isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        if isinstance(elem, int):
            return torch.tensor(batch)
        if isinstance(elem, str):
            return batch
        if isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        if isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        if isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate(self, batch):
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )


def collate_fn(data_list):
    data_list = [x for x in data_list if x is not None]
    return data_list


class DataListLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, List[BaseData]], batch_size: int = 1, shuffle: bool = False, **kwargs):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
