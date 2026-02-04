import os
import glob
import json
import h5py
import pytz
import numpy as np
import torch

from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.datapipe import Datapipe
from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle
from onescience.datapipes.core import BaseDataset

class ERA5Datapipe(Datapipe):
    def __init__(self, params, distributed, output_steps=1, input_steps=1, normalize=True):
        """
        初始化ERA5数据管道
        
        Args:
            params: 配置参数对象，包含数据集和加载器等配置信息
            distributed: 是否使用分布式训练
            output_steps: 输出时间步数，默认为1
            input_steps: 输入时间步数，默认为1
            normalize: 是否对数据进行标准化，默认为True
        """
        self.params = params
        self.dataset = params.dataset
        self.distributed = distributed
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.normalize = normalize
        # 存储数据加载器和采样器
        self.train_loader = None
        self.train_sampler = None
        self.val_loader = None
        self.val_sampler = None
        self.test_loader = None
        # 标记数据加载器是否已初始化
        self._train_loader_initialized = False
        self._val_loader_initialized = False
        self._test_loader_initialized = False

    def train_dataloader(self):
        """
        创建并返回训练数据加载器。

        该方法初始化训练阶段的数据加载器，包括创建ERA5HDF5Dataset实例、配置分布式采样器
        以及设置DataLoader的各种参数。数据加载器会被缓存以供重复使用。

        Returns:
            tuple: 包含两个元素的元组
                - DataLoader: 训练数据加载器，用于批量加载训练数据
                - DistributedSampler or None: 分布式采样器（如果启用分布式训练），否则为None
        """
        # 创建训练数据集实例
        data = ERA5HDF5Dataset(dataset=self.dataset, mode='train', output_steps=self.output_steps, input_steps=self.input_steps, normalize=self.normalize)
        # 创建分布式采样器（如果启用分布式训练）
        sampler = DistributedSampler(data, shuffle=True) if self.distributed else None
        # 创建训练数据加载器
        data_loader = DataLoader(data,
                                 batch_size=self.params.dataloader.batch_size,
                                 drop_last=True if self.distributed else False,
                                 num_workers=self.params.dataloader.num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=sampler)
        # 缓存数据加载器和采样器
        self.train_loader = data_loader
        self.train_sampler = sampler
        self._train_loader_initialized = True
        return data_loader, sampler

    def val_dataloader(self):
        """
        创建验证数据加载器

        创建并配置用于验证阶段的数据加载器，使用ERA5HDF5Dataset数据集，
        支持分布式训练，并缓存数据加载器和采样器以提高性能。

        Returns:
            tuple: (data_loader, sampler) - 验证数据加载器和分布式采样器
        """
        # 创建验证数据集实例
        data = ERA5HDF5Dataset(dataset=self.dataset, mode='val', output_steps=self.output_steps, input_steps=self.input_steps, normalize=self.normalize)
        # 创建分布式采样器（验证集不进行shuffle）
        sampler = DistributedSampler(data, shuffle=False) if self.distributed else None
        # 创建验证数据加载器
        data_loader = DataLoader(data,
                                 batch_size=self.params.dataloader.batch_size,
                                 drop_last=True if self.distributed else False,
                                 num_workers=self.params.dataloader.num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=sampler)
        # 缓存数据加载器和采样器
        self.val_loader = data_loader
        self.val_sampler = sampler
        self._val_loader_initialized = True
        return data_loader, sampler

    def test_dataloader(self):
        """
        创建并返回测试数据加载器。
        
        该方法创建一个用于测试阶段的数据加载器，配置为单样本批次，
        适用于模型验证和评估阶段。
        
        Returns:
            DataLoader: 配置好的测试数据加载器，包含测试数据集，
                       batch_size=1, num_workers=4, pin_memory=True,
                       shuffle=False
        """
        # 创建测试数据集实例
        data = ERA5HDF5Dataset(dataset=self.dataset, mode='test', output_steps=self.output_steps, input_steps=self.input_steps, normalize=self.normalize)
        # 创建测试数据加载器（batch_size固定为1，num_workers固定为4）
        data_loader = DataLoader(data,
                                 batch_size=1,
                                 drop_last=True if self.distributed else False,
                                 num_workers=4,
                                 pin_memory=True,
                                 shuffle=False)
        # 缓存数据加载器
        self.test_loader = data_loader
        self._test_loader_initialized = True
        return data_loader
    
    
class ERA5HDF5Dataset(BaseDataset):
    def __init__(self, dataset, mode='train', output_steps=1, input_steps=1, normalize=True, patch_size=[1, 1]):
        """
        初始化ERA5 HDF5数据集
        
        Args:
            dataset: 数据集配置参数对象，包含数据目录等配置信息
            mode: 数据集模式，可选'train'、'val'或'test'，默认为'train'
            output_steps: 输出时间步数，默认为1
            input_steps: 输入时间步数，默认为1
            normalize: 是否对数据进行标准化，默认为True
            patch_size: 数据块大小，格式为[height, width]，默认为[1, 1]
        """
        self.params = dataset
        self.data_dir = self.params.data_dir
        self.mode = mode
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.patch_size = patch_size
        self.dt = self.params.time_res
        self.normalize = normalize

        self.metadata = None
        self.years = []
        self.variables = []
        self.channel_indices = []
        self.mu = None
        self.sd = None
        self.selected_years = []
        self.files = {}
        self.samples_per_year = 0
        self.total_samples = 0
        self.img_shape = None
        self.latlon_torch = None
        # 数据统计信息
        self.global_means = None
        self.global_stds = None
        # 数据预处理标志
        self._paths_initialized = False
        self._normalization_initialized = False
        self._split_initialized = False
        self._files_initialized = False
        self._latlon_initialized = False
        self._shape_initialized = False

        self._init_paths()
        self._init_normalization()
        self._init_split()
        self._init_files()
        self._init_latlon()
        self._init_shape()
        super().__init__(self.params)


    def _init_paths(self):
        # 初始化数据路径和元数据
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.years = list(map(int, self.metadata["years"]))
        self.variables = self.metadata["variables"]

        # 检查 channels 是否都在 metadata.variables 中
        missing = [ch for ch in self.params.channels if ch not in self.variables]
        if missing:
            raise ValueError(f"❌ Missing required variables in metadata: {missing}")
        
        # 标记路径初始化完成
        self._paths_initialized = True


    def _init_normalization(self):
        # 初始化通道索引
        self.channel_indices = [self.variables.index(v) for v in self.params.channels]
        # 加载全局统计量
        mu = np.load(os.path.join(self.params.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
        std = np.load(os.path.join(self.params.stats_dir, "global_stds.npy"))
        # 提取当前通道的统计量
        self.mu = mu[:, self.channel_indices, :, :]
        self.sd = std[:, self.channel_indices, :, :]
        # 缓存完整的统计信息
        self.global_means = mu
        self.global_stds = std
        
        # 标记标准化初始化完成
        self._normalization_initialized = True


    def _init_split(self):
        y = sorted(self.years)
        tips = False
        error = False
        # 1. use ratio to select data
        if isinstance(self.params.train_ratio, float): 
            if self.params.train_ratio + self.params.val_ratio + self.params.test_ratio > 1:
                error = True
            if self.params.train_ratio + self.params.val_ratio + self.params.test_ratio != 1:
                tips = True
            n_train = int(len(y) * self.params.train_ratio)
            n_val = int(len(y) * self.params.val_ratio)
            year_splits = {
                "train": y[:n_train],
                "val": y[n_train:n_train + n_val],
                "test": y[n_train + n_val:]
            }
        # 2. use number to select years
        if isinstance(self.params.train_ratio, int):
            if self.params.train_ratio + self.params.val_ratio + self.params.test_ratio > len(y):
                error = True
            if self.params.train_ratio + self.params.val_ratio + self.params.test_ratio != len(y):
                tips = True
            n_train =  self.params.train_ratio
            n_val = self.params.val_ratio
            year_splits = {
                "train": y[:n_train],
                "val": y[n_train:n_train + n_val],
                "test": y[n_train + n_val:]
            }
        if isinstance(self.params.train_ratio, (list, tuple, set)):
            self.params.train_ratio = set(self.params.train_ratio)
            self.params.val_ratio = set(self.params.val_ratio)
            self.params.test_ratio = set(self.params.test_ratio)
            if len(self.params.train_ratio)+len(self.params.val_ratio)+len(self.params.test_ratio) > len(y):
                error = True
            if len(self.params.train_ratio)+len(self.params.val_ratio)+len(self.params.test_ratio) != len(y):
                tips = True
            if self.params.train_ratio.issubset(set(self.years)) and self.params.val_ratio.issubset(set(self.years)) and self.params.test_ratio.issubset(set(self.years)):
                year_splits = {
                    "train": self.params.train_ratio,
                    "val": self.params.val_ratio,
                    "test": self.params.test_ratio
                }
            else:
                error = True

        if error:
            print('\n')
            print('-' * 50)
            print(f'❌ ❌ Train/Val/Test settings must use 1.ratio or 2.digital numbers or 3.specific years.')
            print(f'If using ratio, please ensure the sum of all ratios less than 1.')
            print(f'If using digital number, please ensure the sum of number less than total years {len(y)}.')
            print(f'If using specific years, please ensure the years are exist in provided dataset.')
            print(f'We provided {len(y)} years data, which are {y}')
            print(f'❌ ❌ Now settings are train: {self.params.train_ratio}  val: {self.params.val_ratio}  test: {self.params.test_ratio}, please check.')
            print('-' * 50)
            exit()

        if tips:
            print('\n')
            print('-' * 50)
            print(f'⚠️ ⚠️ Current Train/Val/Test settings can use this ERA5 dataset. But you may not use the whole dataset.')
            print(f'⚠️ ⚠️ This is not an error, you can still train the model, or change the config to use whole dataset.')
            print(f'⚠️ ⚠️ We provided {len(y)} years data, which are {y}.')
            print(f'⚠️ ⚠️ Now settings are train: {self.params.train_ratio}  val: {self.params.val_ratio}  test: {self.params.test_ratio}, please ensure.')
            print('-' * 50)
        self.selected_years = list(year_splits[self.mode])
        # 标记分割初始化完成
        self._split_initialized = True
        

    def _init_files(self):
        # 初始化文件路径和样本统计
        for year in self.selected_years:
            path = os.path.join(self.data_dir, 'data', str(year))
            files = sorted(glob.glob(os.path.join(path, "*.h5")))
            self.files[year] = files
        self.samples_per_year = len(files) - self.output_steps - (self.input_steps - 1)
        self.total_samples = len(self.selected_years) * self.samples_per_year
        
        # 标记文件初始化完成
        self._files_initialized = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('\n')
            print('-' * 50)
            print(f"📂 Mode: {self.mode}, used: {self.selected_years} years")
            print(f'📂 each years contains {self.samples_per_year} (Each year contains {len(files)}, input {self.input_steps}, output {self.output_steps})')
            print(f'📂 whole dataset contains {len(self.variables)} variables, this model use {len(self.channel_indices)} variables.')
            print(f'📂 {len(self.selected_years)} years * {self.samples_per_year} samples = Total {len(self.selected_years) * self.samples_per_year} usable samples.')
            print('-' * 50, '\n')


    def _init_latlon(self):
        # 初始化经纬度网格
        latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=self.params.img_size[-2:])
        self.latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)
        
        # 标记经纬度初始化完成
        self._latlon_initialized = True


    def _init_shape(self):
        # 初始化数据形状信息
        sample_file = self.files[self.selected_years[0]][0]
        with h5py.File(sample_file, "r") as f:
            shape = f["fields"].shape  # [N, H, W]
            self.img_shape = [s - s % self.patch_size[i] for i, s in enumerate(shape[-2:])]
            
        # 标记形状初始化完成
        self._shape_initialized = True


    def __len__(self):
        # 返回数据集总样本数
        return self.total_samples


    def __getitem__(self, idx):
        # 计算年份索引和步长索引
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        year = self.selected_years[year_idx]
        files = self.files[year]
        # 计算需要的文件索引范围
        file_indices = range(step_idx, step_idx + self.input_steps + self.output_steps)
        
        # 收集数据和对应的时间索引
        data_list = []
        time_index = []
        for i in file_indices:
            with h5py.File(files[i], "r") as f:
                data = f["fields"][:]  # [N, H, W]
                data = data[self.channel_indices]
                data_list.append(data)
                time_index.append(files[i][-13:-3])

        # 堆叠数据形成时间序列
        data = np.stack(data_list, axis=0)  # [T, N, H, W]
        # 分离输入和输出变量
        invar = torch.as_tensor(data[:self.input_steps])
        outvar = torch.as_tensor(data[self.input_steps:])
        # 根据img_shape裁剪数据
        h, w = self.img_shape
        invar = invar[:, :, :h, :w]
        outvar = outvar[:, :, :h, :w]
        # 数据标准化
        if self.normalize:
            invar = (invar - self.mu) / self.sd
            outvar = (outvar - self.mu) / self.sd
        
        # 计算时间戳和天顶角余弦
        start_time = datetime(year, 1, 1, tzinfo=pytz.utc)
        timestamps = np.array([(start_time + timedelta(hours=(step_idx + t) * self.dt)).timestamp()
                               for t in range(self.output_steps)])
        timestamps = torch.from_numpy(timestamps)
        cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon_torch).float()

        # 返回处理后的数据
        return invar.squeeze(0), outvar.squeeze(0), cos_zenith, step_idx, time_index