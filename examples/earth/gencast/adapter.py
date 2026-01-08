"""
GenCast 数据适配层
将 PyTorch DataLoader 输出转换为 GenCast 期望的 xarray.Dataset 格式
"""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import torch
import os


class GenCastAdapter:
    """
    将 PyTorch DataLoader 输出转换为 GenCast 期望的 xarray.Dataset 格式
    """
    
    def __init__(
        self,
        channels: List[str],
        surface_variables: List[str],
        pressure_level_variables: List[str],
        pressure_levels: List[int],
        target_only_variables: List[str],
        static_variables: Dict[str, np.ndarray],
        lat: np.ndarray,
        lon: np.ndarray,
        time_interval_hours: int = 6,
    ):
        """
        Args:
            channels: DataLoader 返回的通道名列表，顺序与 invar/outvar 一致
            surface_variables: surface 变量名列表 (无 level 维度)
            pressure_level_variables: pressure level 变量名列表 (有 level 维度)
            pressure_levels: 气压层列表，如 [50, 100, ..., 1000]
            target_only_variables: 仅在 target 中出现的变量 (如 total_precipitation)
            static_variables: 静态变量字典 {'geopotential_at_surface': array, 'land_sea_mask': array}
            lat: 纬度数组
            lon: 经度数组
            time_interval_hours: 时间间隔（小时）
        """
        self.channels = channels
        self.surface_variables = surface_variables
        self.pressure_level_variables = pressure_level_variables
        self.pressure_levels = pressure_levels
        self.target_only_variables = target_only_variables
        self.static_variables = static_variables
        self.lat = lat
        self.lon = lon
        self.time_interval_hours = time_interval_hours
        
        # 构建 channel 索引映射
        self._build_channel_index_map()
    
    def _build_channel_index_map(self):
        """构建从变量名到 channel 索引的映射"""
        self.channel_to_idx = {ch: i for i, ch in enumerate(self.channels)}
        
        # Surface 变量索引
        self.surface_indices = {}
        for var in self.surface_variables:
            if var in self.channel_to_idx:
                self.surface_indices[var] = self.channel_to_idx[var]
        
        # Pressure level 变量索引 (每个变量对应多个 level)
        self.pressure_level_indices = {}
        for var in self.pressure_level_variables:
            indices = []
            for level in self.pressure_levels:
                ch_name = f"{var}_{level}"
                if ch_name in self.channel_to_idx:
                    indices.append(self.channel_to_idx[ch_name])
            if indices:
                self.pressure_level_indices[var] = indices
    
    def _parse_datetime(self, time_str: str) -> datetime:
        """解析时间字符串 '2020020206' -> datetime"""
        return datetime.strptime(str(time_str), "%Y%m%d%H")
    
    def _compute_seconds_since_epoch(self, dt: datetime) -> float:
        """计算从 epoch 到指定时间的秒数"""
        epoch = datetime(1970, 1, 1)
        return (dt - epoch).total_seconds()
    
    def _compute_year_progress(self, seconds_since_epoch: np.ndarray) -> np.ndarray:
        """计算年进度 [0, 1)"""
        SEC_PER_DAY = 3600 * 24
        AVG_DAY_PER_YEAR = 365.24219
        AVG_SEC_PER_YEAR = SEC_PER_DAY * AVG_DAY_PER_YEAR
        years_since_epoch = seconds_since_epoch / AVG_SEC_PER_YEAR
        return np.mod(years_since_epoch, 1.0).astype(np.float32)
    
    def _compute_day_progress(self, seconds_since_epoch: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """计算日进度，考虑经度偏移"""
        SEC_PER_DAY = 3600 * 24
        day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY
        longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
        day_progress = np.mod(
            day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
        )
        return day_progress.astype(np.float32)
    
    def _extract_surface_data(self, data: np.ndarray, var_name: str) -> Optional[np.ndarray]:
        """
        从数据中提取 surface 变量
        
        Args:
            data: shape [B, T, C, H, W] 或 [B, C, H, W]
            var_name: 变量名
            
        Returns:
            shape [B, T, H, W] 或 None
        """
        if var_name not in self.surface_indices:
            return None
        idx = self.surface_indices[var_name]
        
        if data.ndim == 5:  # [B, T, C, H, W]
            return data[:, :, idx, :, :]
        else:  # [B, C, H, W]
            return data[:, np.newaxis, idx, :, :]  # 添加 time 维度
    
    def _extract_pressure_level_data(self, data: np.ndarray, var_name: str) -> Optional[np.ndarray]:
        """
        从数据中提取 pressure level 变量
        
        Args:
            data: shape [B, T, C, H, W] 或 [B, C, H, W]
            var_name: 变量名
            
        Returns:
            shape [B, T, level, H, W] 或 None
        """
        if var_name not in self.pressure_level_indices:
            return None
        indices = self.pressure_level_indices[var_name]
        
        if data.ndim == 5:  # [B, T, C, H, W]
            level_data = data[:, :, indices, :, :]  # [B, T, level, H, W]
        else:  # [B, C, H, W]
            level_data = data[:, indices, :, :]  # [B, level, H, W]
            level_data = level_data[:, np.newaxis, :, :, :]  # [B, 1, level, H, W]
        
        return level_data
    
    def _create_time_deltas(self, num_times: int, is_input: bool) -> np.ndarray:
        """
        创建时间增量数组
        
        Args:
            num_times: 时间步数
            is_input: 是否是输入（True: 负增量到0，False: 正增量）
            
        Returns:
            timedelta64 数组
        """
        interval_ns = self.time_interval_hours * 3600 * 1e9  # 转换为纳秒
        
        if is_input:
            # 输入: [..., -interval, 0]
            deltas = np.array([
                (i - num_times + 1) * interval_ns for i in range(num_times)
            ], dtype='timedelta64[ns]')
        else:
            # 输出: [interval, 2*interval, ...]
            deltas = np.array([
                (i + 1) * interval_ns for i in range(num_times)
            ], dtype='timedelta64[ns]')
        
        return deltas
    
    def convert(
        self,
        invar: torch.Tensor,
        outvar: torch.Tensor,
        time_index: List[str],
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """
        将 DataLoader 输出转换为 GenCast 格式
        
        Args:
            invar: 输入数据 [B, T, C, H, W]
            outvar: 输出数据 [B, C, H, W] 或 [B, T, C, H, W]
            time_index: 时间戳列表，长度 = input_steps + output_steps
            
        Returns:
            inputs: xarray.Dataset
            targets: xarray.Dataset
            forcings: xarray.Dataset
        """
        # 转换为 numpy
        if isinstance(invar, torch.Tensor):
            invar = invar.cpu().numpy()
        if isinstance(outvar, torch.Tensor):
            outvar = outvar.cpu().numpy()
        
        # 确保 outvar 有 time 维度
        if outvar.ndim == 4:  # [B, C, H, W]
            outvar = outvar[:, np.newaxis, :, :, :]  # [B, 1, C, H, W]
        
        batch_size = invar.shape[0]
        input_times = invar.shape[1]
        output_times = outvar.shape[1]
        
        # 解析时间戳
        datetimes = [self._parse_datetime(t) for t in time_index]
        input_datetimes = datetimes[:input_times]
        target_datetimes = datetimes[input_times:]
        
        # 创建坐标
        input_time_deltas = self._create_time_deltas(input_times, is_input=True)
        target_time_deltas = self._create_time_deltas(output_times, is_input=False)
        
        coords_base = {
            'batch': np.arange(batch_size),
            'lat': self.lat,
            'lon': self.lon,
            'level': np.array(self.pressure_levels, dtype=np.int32),
        }
        
        # ========== 构建 inputs Dataset ==========
        input_data_vars = {}
        
        # Surface 变量 (排除 target_only)
        for var in self.surface_variables:
            if var in self.target_only_variables:
                continue
            data = self._extract_surface_data(invar, var)
            if data is not None:
                input_data_vars[var] = (['batch', 'time', 'lat', 'lon'], data)
        
        # Pressure level 变量
        for var in self.pressure_level_variables:
            data = self._extract_pressure_level_data(invar, var)
            if data is not None:
                input_data_vars[var] = (['batch', 'time', 'level', 'lat', 'lon'], data)
        
        # 计算 input 时刻的 forcing
        input_seconds = np.array([
            self._compute_seconds_since_epoch(dt) for dt in input_datetimes
        ])[np.newaxis, :]  # [1, time]
        input_seconds = np.repeat(input_seconds, batch_size, axis=0)  # [batch, time]
        
        year_progress = self._compute_year_progress(input_seconds)
        input_data_vars['year_progress_sin'] = (['batch', 'time'], np.sin(2 * np.pi * year_progress).astype(np.float32))
        input_data_vars['year_progress_cos'] = (['batch', 'time'], np.cos(2 * np.pi * year_progress).astype(np.float32))
        
        day_progress = self._compute_day_progress(input_seconds, self.lon)  # [batch, time, lon]
        input_data_vars['day_progress_sin'] = (['batch', 'time', 'lon'], np.sin(2 * np.pi * day_progress).astype(np.float32))
        input_data_vars['day_progress_cos'] = (['batch', 'time', 'lon'], np.cos(2 * np.pi * day_progress).astype(np.float32))
        
        # 静态变量
        for static_name, static_data in self.static_variables.items():
            input_data_vars[static_name] = (['lat', 'lon'], static_data)
        
        coords_input = {**coords_base, 'time': input_time_deltas}
        inputs = xr.Dataset(input_data_vars, coords=coords_input)
        
        # ========== 构建 targets Dataset ==========
        target_data_vars = {}
        
        # Surface 变量 (包括 target_only)
        for var in self.surface_variables:
            data = self._extract_surface_data(outvar, var)
            if data is not None:
                target_data_vars[var] = (['batch', 'time', 'lat', 'lon'], data)
        
        # Pressure level 变量
        for var in self.pressure_level_variables:
            data = self._extract_pressure_level_data(outvar, var)
            if data is not None:
                target_data_vars[var] = (['batch', 'time', 'level', 'lat', 'lon'], data)
        
        coords_target = {**coords_base, 'time': target_time_deltas}
        targets = xr.Dataset(target_data_vars, coords=coords_target)
        
        # ========== 构建 forcings Dataset ==========
        target_seconds = np.array([
            self._compute_seconds_since_epoch(dt) for dt in target_datetimes
        ])[np.newaxis, :]  # [1, time]
        target_seconds = np.repeat(target_seconds, batch_size, axis=0)  # [batch, time]
        
        year_progress_target = self._compute_year_progress(target_seconds)
        day_progress_target = self._compute_day_progress(target_seconds, self.lon)
        
        forcing_data_vars = {
            'year_progress_sin': (['batch', 'time'], np.sin(2 * np.pi * year_progress_target).astype(np.float32)),
            'year_progress_cos': (['batch', 'time'], np.cos(2 * np.pi * year_progress_target).astype(np.float32)),
            'day_progress_sin': (['batch', 'time', 'lon'], np.sin(2 * np.pi * day_progress_target).astype(np.float32)),
            'day_progress_cos': (['batch', 'time', 'lon'], np.cos(2 * np.pi * day_progress_target).astype(np.float32)),
        }
        
        coords_forcing = {
            'batch': np.arange(batch_size),
            'time': target_time_deltas,
            'lon': self.lon,
        }
        forcings = xr.Dataset(forcing_data_vars, coords=coords_forcing)
        
        return inputs, targets, forcings


def create_adapter(
    channels: List[str],
    surface_variables: List[str],
    pressure_level_variables: List[str],
    pressure_levels: List[int],
    target_only_variables: List[str],
    static_dir: str,
    static_files: Dict[str, str],
    img_size: Tuple[int, int],
    time_interval_hours: int = 6,
) -> GenCastAdapter:
    """
    创建适配器的工厂函数
    
    Args:
        channels: 通道名列表
        surface_variables: surface 变量列表
        pressure_level_variables: pressure level 变量列表
        pressure_levels: 气压层列表
        target_only_variables: 仅在 target 中的变量
        static_dir: 静态文件目录
        static_files: 静态文件名映射 {'geopotential_at_surface': 'topography.npy', ...}
        img_size: 图像尺寸 (H, W)
        time_interval_hours: 时间间隔
        
    Returns:
        GenCastAdapter 实例
    """
    # 加载静态变量
    static_variables = {}
    for var_name, file_name in static_files.items():
        file_path = os.path.join(static_dir, file_name)
        if os.path.exists(file_path):
            static_variables[var_name] = np.load(file_path).astype(np.float32)
            print(f"Loaded static variable: {var_name} from {file_path}, shape: {static_variables[var_name].shape}")
        else:
            print(f"Warning: Static file not found: {file_path}")
    
    # 生成坐标
    lat = np.linspace(90, -90, img_size[0]).astype(np.float32)
    lon = np.linspace(0, 360 - 360/img_size[1], img_size[1]).astype(np.float32)
    
    return GenCastAdapter(
        channels=channels,
        surface_variables=surface_variables,
        pressure_level_variables=pressure_level_variables,
        pressure_levels=pressure_levels,
        target_only_variables=target_only_variables,
        static_variables=static_variables,
        lat=lat,
        lon=lon,
        time_interval_hours=time_interval_hours,
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import sys
    import os
    
    # 添加项目路径（根据你的实际路径调整）
    current_path = os.getcwd()
    sys.path.append(current_path)
    
    from onescience.datapipes.climate import ERA5Datapipe
    from onescience.utils.YParams import YParams
    
    print("=" * 60)
    print("GenCast Adapter 测试")
    print("=" * 60)
    
    # ========== 读取配置 ==========
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_gencast = YParams(config_file_path, "gencast")
    
    print(f"\n>>> 配置信息:")
    print(f"  img_size: {cfg_data.dataset.img_size}")
    print(f"  time_res: {cfg_data.dataset.time_res}")
    print(f"  channels 数量: {len(cfg_data.dataset.channels)}")
    print(f"  surface_variables: {cfg_gencast.surface_variables}")
    print(f"  pressure_level_variables: {cfg_gencast.pressure_level_variables}")
    print(f"  pressure_levels: {cfg_gencast.pressure_levels}")
    
    # ========== 创建 DataLoader ==========
    # 注意：测试时不使用分布式
    datapipe = ERA5Datapipe(
        params=cfg_data, 
        distributed=False,
        input_steps=2,
        output_steps=1,
    )
    train_dataloader, _ = datapipe.train_dataloader()
    
    print(f"\n>>> DataLoader 信息:")
    print(f"  batch_size: {cfg_data.dataloader.batch_size}")
    print(f"  total batches: {len(train_dataloader)}")
    
    # ========== 加载静态变量 ==========
    static_dir = cfg_data.dataset.static_dir
    static_files = cfg_gencast.static_files
    
    static_variables = {}
    for var_name, file_name in static_files.items():
        file_path = os.path.join(static_dir, file_name)
        if os.path.exists(file_path):
            data = np.load(file_path).astype(np.float32)
            # 如果分辨率不匹配，需要 resize（这里假设匹配）
            static_variables[var_name] = data
            print(f"  Loaded {var_name}: shape={data.shape}")
        else:
            print(f"  Warning: {file_path} not found")
    
    # ========== 创建适配器 ==========
    adapter = create_adapter(
        channels=cfg_data.dataset.channels,
        surface_variables=cfg_gencast.surface_variables,
        pressure_level_variables=cfg_gencast.pressure_level_variables,
        pressure_levels=cfg_gencast.pressure_levels,
        target_only_variables=cfg_gencast.target_only_variables,
        static_dir=cfg_data.dataset.static_dir,
        static_files=cfg_gencast.static_files,
        img_size=cfg_data.dataset.img_size,
        time_interval_hours=cfg_data.dataset.time_res,
    )
    
    # ========== 获取一个 batch 进行测试 ==========
    print("\n" + "-" * 60)
    print("从 DataLoader 获取数据并测试转换")
    print("-" * 60)
    
    for batch_idx, data in enumerate(train_dataloader):
        invar = data[0]
        outvar = data[1]
        # cos_zenith = data[2]  # 本模型不需要
        # step_idx = data[3]    # 本模型不需要
        time_index = data[4]
        
        print(f"\n>>> Batch {batch_idx} 原始数据:")
        print(f"  invar shape: {invar.shape}")
        print(f"  outvar shape: {outvar.shape}")
        print(f"  time_index: {time_index}")
        
        # 执行转换
        inputs, targets, forcings = adapter.convert(invar, outvar, time_index)
        
        # 打印 inputs 结构
        print("\n>>> 转换后 inputs:")
        print(f"  dims: {dict(inputs.dims)}")
        print(f"  data_vars ({len(inputs.data_vars)}): {list(inputs.data_vars)}")
        for var in list(inputs.data_vars)[:3]:  # 只打印前3个
            v = inputs[var]
            print(f"    {var}: dims={v.dims}, shape={v.shape}")
        print(f"    ... (共 {len(inputs.data_vars)} 个变量)")
        
        # 打印 targets 结构
        print("\n>>> 转换后 targets:")
        print(f"  dims: {dict(targets.dims)}")
        print(f"  data_vars ({len(targets.data_vars)}): {list(targets.data_vars)}")
        for var in list(targets.data_vars)[:3]:
            v = targets[var]
            print(f"    {var}: dims={v.dims}, shape={v.shape}")
        print(f"    ... (共 {len(targets.data_vars)} 个变量)")
        
        # 打印 forcings 结构
        print("\n>>> 转换后 forcings:")
        print(f"  dims: {dict(forcings.dims)}")
        print(f"  data_vars: {list(forcings.data_vars)}")
        for var in forcings.data_vars:
            v = forcings[var]
            print(f"    {var}: dims={v.dims}, shape={v.shape}")
        
        # 打印坐标
        print("\n>>> 坐标详情:")
        print(f"  lat: [{inputs.coords['lat'].values[0]:.1f}, ..., {inputs.coords['lat'].values[-1]:.1f}], len={len(inputs.coords['lat'])}")
        print(f"  lon: [{inputs.coords['lon'].values[0]:.1f}, ..., {inputs.coords['lon'].values[-1]:.1f}], len={len(inputs.coords['lon'])}")
        print(f"  level: {list(inputs.coords['level'].values)}")
        print(f"  input time: {list(inputs.coords['time'].values)}")
        print(f"  target time: {list(targets.coords['time'].values)}")
        
        # 只测试一个 batch
        break
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)