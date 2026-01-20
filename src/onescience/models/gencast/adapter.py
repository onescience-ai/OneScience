import numpy as np
import xarray as xr
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import torch
import os
from scipy.ndimage import zoom


class GenCastAdapter:
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
        target_shape: Tuple[int, int] = None,
    ):
        self.channels = channels
        self.surface_variables = surface_variables
        self.pressure_level_variables = pressure_level_variables
        self.pressure_levels = pressure_levels
        self.target_only_variables = target_only_variables
        self.static_variables = static_variables
        self.lat = lat
        self.lon = lon
        self.time_interval_hours = time_interval_hours
        self.target_shape = target_shape  
        self._build_channel_index_map()
    
    def _build_channel_index_map(self):
        self.channel_to_idx = {ch: i for i, ch in enumerate(self.channels)}

        self.surface_indices = {}
        for var in self.surface_variables:
            if var in self.channel_to_idx:
                self.surface_indices[var] = self.channel_to_idx[var]

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
        return datetime.strptime(str(time_str), "%Y%m%d%H")
    
    def _compute_seconds_since_epoch(self, dt: datetime) -> float:
        epoch = datetime(1970, 1, 1)
        return (dt - epoch).total_seconds()
    
    def _compute_year_progress(self, seconds_since_epoch: np.ndarray) -> np.ndarray:
        SEC_PER_DAY = 3600 * 24
        AVG_DAY_PER_YEAR = 365.24219
        AVG_SEC_PER_YEAR = SEC_PER_DAY * AVG_DAY_PER_YEAR
        years_since_epoch = seconds_since_epoch / AVG_SEC_PER_YEAR
        return np.mod(years_since_epoch, 1.0).astype(np.float32)
    
    def _compute_day_progress(self, seconds_since_epoch: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        SEC_PER_DAY = 3600 * 24
        day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY
        longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
        day_progress = np.mod(
            day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
        )
        return day_progress.astype(np.float32)
    
    def _extract_surface_data(self, data: np.ndarray, var_name: str) -> Optional[np.ndarray]:
        if var_name not in self.surface_indices:
            return None
        idx = self.surface_indices[var_name]
        
        if data.ndim == 5:  # [B, T, C, H, W]
            return data[:, :, idx, :, :]
        else:  # [B, C, H, W]
            return data[:, np.newaxis, idx, :, :]  #
    
    def _extract_pressure_level_data(self, data: np.ndarray, var_name: str) -> Optional[np.ndarray]:
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
        interval_ns = int(self.time_interval_hours * 3600 * 1e9)  
        
        if is_input:
            deltas = [np.timedelta64((i - num_times + 1) * interval_ns, 'ns') for i in range(num_times)]
        else:
            deltas = [np.timedelta64((i + 1) * interval_ns, 'ns') for i in range(num_times)]
        return np.array(deltas)
    
    def convert(
        self,
        invar: torch.Tensor,
        outvar: torch.Tensor,
        time_index: List,
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:

        if isinstance(invar, torch.Tensor):
            invar = invar.cpu().numpy()
        if isinstance(outvar, torch.Tensor):
            outvar = outvar.cpu().numpy()
        if isinstance(time_index[0], (list, tuple)):
            time_index = [t[0] for t in time_index]
        if outvar.ndim == 4:  # [B, C, H, W]
            outvar = outvar[:, np.newaxis, :, :, :]  # [B, 1, C, H, W]

        invar = np.flip(invar, axis=3).copy()
        outvar = np.flip(outvar, axis=3).copy()

        if self.target_shape is not None:
            target_h, target_w = self.target_shape
            invar = downsample_to_shape(invar, target_h, target_w, axis_h=3, axis_w=4)
            outvar = downsample_to_shape(outvar, target_h, target_w, axis_h=3, axis_w=4)
        # ======================================
            
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
        
        input_data_vars = {}
        for var in self.surface_variables:
            if var in self.target_only_variables:
                continue
            data = self._extract_surface_data(invar, var)
            if data is not None:
                input_data_vars[var] = (['batch', 'time', 'lat', 'lon'], data)

        for var in self.pressure_level_variables:
            data = self._extract_pressure_level_data(invar, var)
            if data is not None:
                input_data_vars[var] = (['batch', 'time', 'level', 'lat', 'lon'], data)
        
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
        
        for static_name, static_data in self.static_variables.items():
            input_data_vars[static_name] = (['lat', 'lon'], static_data)
        
        coords_input = {**coords_base, 'time': input_time_deltas}
        inputs = xr.Dataset(input_data_vars, coords=coords_input)
        
        target_data_vars = {}

        for var in self.surface_variables:
            data = self._extract_surface_data(outvar, var)
            if data is not None:
                target_data_vars[var] = (['batch', 'time', 'lat', 'lon'], data)

        for var in self.pressure_level_variables:
            data = self._extract_pressure_level_data(outvar, var)
            if data is not None:
                target_data_vars[var] = (['batch', 'time', 'level', 'lat', 'lon'], data)
        
        coords_target = {**coords_base, 'time': target_time_deltas}
        targets = xr.Dataset(target_data_vars, coords=coords_target)

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
    downsample_factor: int = 1,
    target_shape: Tuple[int, int] = None,
) -> GenCastAdapter:
    if target_shape is not None:
        new_h, new_w = target_shape
    elif downsample_factor > 1:
        new_h = img_size[0] // downsample_factor
        new_w = img_size[1] // downsample_factor
    else:
        new_h, new_w = img_size

    static_variables = {}
    for var_name, file_name in static_files.items():
        file_path = os.path.join(static_dir, file_name)
        if os.path.exists(file_path):
            data = np.load(file_path).astype(np.float32)
            data = np.flip(data, axis=0).copy()
            if (data.shape[0], data.shape[1]) != (new_h, new_w):
                zoom_h = new_h / data.shape[0]
                zoom_w = new_w / data.shape[1]
                data = zoom(data, (zoom_h, zoom_w), order=1).astype(np.float32)
            static_variables[var_name] = data
            print(f"Loaded static variable: {var_name} from {file_path}, shape: {data.shape}")
        else:
            print(f"Warning: Static file not found: {file_path}")

    lat = np.linspace(-90, 90, new_h).astype(np.float32)
    lon = np.linspace(0, 360 - 360/new_w, new_w).astype(np.float32)
    
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
        target_shape=(new_h, new_w), 
    )


def downsample_to_shape(data: np.ndarray, target_h: int, target_w: int, axis_h: int = -2, axis_w: int = -1) -> np.ndarray:

    current_h = data.shape[axis_h]
    current_w = data.shape[axis_w]
    
    if current_h == target_h and current_w == target_w:
        return data

    zoom_factors = [1.0] * data.ndim
    zoom_factors[axis_h] = target_h / current_h
    zoom_factors[axis_w] = target_w / current_w
    
    return zoom(data, zoom_factors, order=1).astype(data.dtype)
