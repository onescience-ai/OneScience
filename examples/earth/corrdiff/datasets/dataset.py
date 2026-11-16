from typing import Iterable, Tuple, Union
import copy
import torch

from onescience.utils.generative import InfiniteSampler
from onescience.distributed import DistributedManager

from . import base, cwb
# , hrrrmini


# this maps all known dataset types to the corresponding init function
# known_datasets = {"cwb": cwb.get_zarr_dataset, "hrrr_mini": hrrrmini.HRRRMiniDataset}
known_datasets = {"cwb": cwb.get_zarr_dataset}

# 主初始化函数
def init_train_valid_datasets_from_config(
    dataset_cfg: dict,                                    # dataset/cwb_train.yaml
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,                                  # 每个数据批次的样本数量
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,     # validation/cwb.yaml
    train_test_split: bool = True,                        
) -> Tuple[
    base.DownscalingDataset,                # 数据集
    Iterable,                               # 数据集迭代器
    Union[base.DownscalingDataset, None],   # 验证集        if train_test_split is True
    Union[Iterable, None],                  # 验证集迭代器
]:
    """
    A wrapper function for managing the train-test split for the CWB dataset.

    Parameters:
    - dataset_cfg (dict): Configuration for the dataset.
    - dataloader_cfg (dict, optional): Configuration for the dataloader. Defaults to None.
    - batch_size (int): The number of samples in each batch of data. Defaults to 1.
    - seed (int): The random seed for dataset shuffling. Defaults to 0.
    - train_test_split (bool): A flag to determine whether to create a validation dataset. Defaults to True.

    Returns:
    - Tuple[base.DownscalingDataset, Iterable, Optional[base.DownscalingDataset], Optional[Iterable]]
    A tuple containing the training dataset and iterator, and optionally the validation dataset and iterator if train_test_split is True.
    """

    config = copy.deepcopy(dataset_cfg)
    (dataset, dataset_iter) = init_dataset_from_config(config, dataloader_cfg, batch_size=batch_size, seed=seed)

    if train_test_split:
        valid_dataset_cfg = copy.deepcopy(config)
        if validation_dataset_cfg:
            valid_dataset_cfg.update(validation_dataset_cfg)
        (valid_dataset, valid_dataset_iter) = init_dataset_from_config(valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed)
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter

# 数据集初始化函数
def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:          # 返回数据集和数据集迭代器
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "cwb")
    if "train_test_split" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["train_test_split"]
    dataset_init_func = known_datasets[dataset_type]    # cwb.get_zarr_dataset

    dataset_obj = dataset_init_func(**dataset_cfg)      # 调用cwb.get_zarr_dataset
    if dataloader_cfg is None:
        dataloader_cfg = {}

    dist = DistributedManager()
    dataset_sampler = InfiniteSampler(dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed) 

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return (dataset_obj, dataset_iterator)
