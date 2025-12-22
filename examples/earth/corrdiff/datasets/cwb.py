"""Streaming images and labels from datasets created with dataset_tool.py.
    _ZarrDataset读取zarr格式数据, 返回数据集对象
    FilterTime从时间维度上划分数据集为训练集和验证集
    ZarrDataset从通道、经纬度维度上对数据进行处理
"""

import logging
import random

import cftime
import cv2
from hydra.utils import to_absolute_path
import numpy as np
import zarr

from .base import ChannelMetadata, DownscalingDataset
from .img_utils import reshape_fields
from .norm import denormalize, normalize

logger = logging.getLogger(__file__)


def get_target_normalizations_v1(group):
    """Get target normalizations using center and scale values from the 'group'.使用group中的中心值和缩放值获取目标的标准化参数"""
    return group["cwb_center"][:], group["cwb_scale"][:]    # cwb数据集4个变量的中心值，4个变量的缩放值，Shape: (4,)


def get_target_normalizations_v2(group):
    """Change the normalizations of the non-gaussian output variables 更改非高斯分布输出变量的标准化参数"""
    center = group["cwb_center"]        # 标准化的中心值
    scale = group["cwb_scale"]          # 标准化的缩放值
    variable = group["cwb_variable"]    

    center = np.where(variable == "maximum_radar_reflectivity", 25.0, center)
    center = np.where(variable == "eastward_wind_10m", 0.0, center)
    center = np.where(variable == "northward_wind_10m", 0, center)

    scale = np.where(variable == "maximum_radar_reflectivity", 25.0, scale)
    scale = np.where(variable == "eastward_wind_10m", 20.0, scale)
    scale = np.where(variable == "northward_wind_10m", 20.0, scale)
    return center, scale


class _ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file

    This dataset should not be modified to add image processing contributions.
    读取zarr格式数据, 返回数据集对象
    """

    path: str

    def __init__(self, path: str, get_target_normalization=get_target_normalizations_v1):
        self.path = path
        self.group = zarr.open_consolidated(path)   # group表示cwa_dataset.zarr
        self.get_target_normalization = get_target_normalization

        # valid indices
        cwb_valid = self.group["cwb_valid"]     # Shape: (35064,)，array([1, 1, 1, ..., 1, 1, 1], dtype=int8)
        era5_valid = self.group["era5_valid"]   # Shape: (35064, 20)，全1
        if not (
            era5_valid.ndim == 2
            and cwb_valid.ndim == 1
            and cwb_valid.shape[0] == era5_valid.shape[0]
        ):
            raise ValueError("Invalid dataset shape")
        era5_all_channels_valid = np.all(era5_valid, axis=-1)   # era5_all_channels_valid.shape=(35064,)，全True
        valid_times = cwb_valid & era5_all_channels_valid       
        # need to cast to bool since cwb_valid is stored as an int8 type in zarr.
        self.valid_times = valid_times != 0                     # self.valid_times.shape=(35064,)，全True

        logger.info("Number of valid times: %d", len(self.valid_times))         # 35064
        # logger.info("input_channels:%s", self.input_channels())                 # 20
        # logger.info("output_channels:%s", self.output_channels())               # 4

    def _get_valid_time_index(self, idx):
        time_indexes = np.arange(self.group["time"].size)           
        if not self.valid_times.dtype == np.bool_:
            raise ValueError("valid_times must be a boolean array")
        valid_time_indexes = time_indexes[self.valid_times]     # 0~35063的数组  0~26303为训练集，26304~35063为验证集
        return valid_time_indexes[idx]

    def __getitem__(self, idx):
        idx_to_load = self._get_valid_time_index(idx)
        target = self.group["cwb"][idx_to_load]     # shape=(4,450,450)
        input = self.group["era5"][idx_to_load]     # shape=(20,450,450)
        label = 0

        # 标准化
        target = self.normalize_output(target[None, ...])[0]
        input = self.normalize_input(input[None, ...])[0]

        return target, input, label

    def longitude(self):
        """The longitude. useful for plotting"""
        return self.group["XLONG"]  # <zarr.core.Array '/XLONG' (450, 450) float32>

    def latitude(self):
        """The latitude. useful for plotting"""
        return self.group["XLAT"]   # <zarr.core.Array '/XLAT' (450, 450) float32>

    def _get_channel_meta(self, variable, level):
        if np.isnan(level):         # 对于地表变量，其level值设为的nan
            level = ""
        return ChannelMetadata(name=variable, level=str(level))

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        variable = self.group["era5_variable"]
        level = self.group["era5_pressure"]
        return [self._get_channel_meta(*v) for v in zip(variable, level)]   # 20

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        variable = self.group["cwb_variable"]
        level = self.group["cwb_pressure"]
        return [self._get_channel_meta(*v) for v in zip(variable, level)]   # 4

    def _read_time(self):
        """The vector of time coordinate has length (self)"""
        return cftime.num2date(
            self.group["time"], units=self.group["time"].attrs["units"]
        )

    def time(self):
        """The vector of time coordinate has length (self)"""
        time = self._read_time()
        return time[self.valid_times].tolist()  
    # [cftime.DatetimeGregorian(2018, 1, 1, 0, 0, 0, 0, has_year_zero=False), ..., cftime.DatetimeGregorian(2021, 12, 31, 23, 0, 0, 0, has_year_zero=False)]

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self.group["cwb"].shape[-2:] # (450,450)

    def _select_norm_channels(self, means, stds, channels):
        if channels is not None:
            means = means[channels]
            stds = stds[channels]
        return (means, stds)  # norm=(means, stds)

    def normalize_input(self, x, channels=None):
        """将物理单位的输入转换为标准化数据。channels表示对指定channels进行转换"""
        norm = self._select_norm_channels(self.group["era5_center"], self.group["era5_scale"], channels)
        return normalize(x, *norm)

    def denormalize_input(self, x, channels=None):
        """将标准化的输入转换为物理单位的数据。"""
        norm = self._select_norm_channels(self.group["era5_center"], self.group["era5_scale"], channels)
        return denormalize(x, *norm)

    def normalize_output(self, x, channels=None):
        """将物理单位的输出转换为标准化数据。"""
        norm = self.get_target_normalization(self.group)
        norm = self._select_norm_channels(*norm, channels)
        return normalize(x, *norm)

    def denormalize_output(self, x, channels=None):
        """将标准化的输出转换为物理单位的数据。"""
        norm = self.get_target_normalization(self.group)
        norm = self._select_norm_channels(*norm, channels)
        return denormalize(x, *norm)

    def info(self):
        return {
            "target_normalization": self.get_target_normalization(self.group),
            "input_normalization": (
                self.group["era5_center"][:],
                self.group["era5_scale"][:],
            ),
        }

    def __len__(self):
        return self.valid_times.sum()


class FilterTime(DownscalingDataset):
    """Filter a time dependent dataset"""

    def __init__(self, dataset, filter_fn):
        """
        Args:
            filter_fn: if filter_fn(time) is True then return point
        """
        self._dataset = dataset
        self._filter_fn = filter_fn
        self._indices = [i for i, t in enumerate(self._dataset.time()) if filter_fn(t)] # self._indices即下标列表，元素为0~26303(训练集)或26304~35063（验证集）
        # cftime.DatetimeGregorian(2018, 1, 1, 10, 0, 0, 0, has_year_zero=False).year=2018

    def longitude(self):
        """Get longitude values from the dataset."""
        return self._dataset.longitude()

    def latitude(self):
        """Get latitude values from the dataset."""
        return self._dataset.latitude()

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        return self._dataset.input_channels()

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        return self._dataset.output_channels()

    def time(self):
        """Get time values from the dataset."""
        time = self._dataset.time()
        return [time[i] for i in self._indices]
    # [cftime.DatetimeGregorian(2018, 1, 1, 0, 0, 0, 0, has_year_zero=False), ..., cftime.DatetimeGregorian(2020, 12, 31, 23, 0, 0, 0, has_year_zero=False)]
    # 或[cftime.DatetimeGregorian(2021, 1, 1, 0, 0, 0, 0, has_year_zero=False), ..., cftime.DatetimeGregorian(2021, 12, 31, 23, 0, 0, 0, has_year_zero=False)]
    def info(self):
        """Get information about the dataset."""
        return self._dataset.info()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self._dataset.image_shape()

    def normalize_input(self, x, channels=None):
        """Convert input from physical units to normalized data."""
        return self._dataset.normalize_input(x, channels=channels)

    def denormalize_input(self, x, channels=None):
        """Convert input from normalized data to physical units."""
        return self._dataset.denormalize_input(x, channels=channels)

    def normalize_output(self, x, channels=None):
        """Convert output from physical units to normalized data."""
        return self._dataset.normalize_output(x, channels=channels)

    def denormalize_output(self, x, channels=None):
        """Convert output from normalized data to physical units."""
        return self._dataset.denormalize_output(x, channels=channels)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def __len__(self):
        return len(self._indices)


def is_2021(time):
    """Check if the given time is in the year 2021."""
    return time.year == 2021


def is_not_2021(time):
    """Check if the given time is not in the year 2021."""
    return not is_2021(time)


class ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file with the
    following schema::

        xarray.Dataset {
        dimensions:
                south_north = 450 ;
                west_east = 450 ;
                west_east_stag = 451 ;
                south_north_stag = 451 ;
                time = 8760 ;
                cwb_channel = 20 ;
                era5_channel = 20 ;

        variables:
                float32 XLAT(south_north, west_east) ;
                        XLAT:FieldType = 104 ;
                        XLAT:MemoryOrder = XY  ;
                        XLAT:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT:stagger =  ;
                        XLAT:units = degree_north ;
                float32 XLAT_U(south_north, west_east_stag) ;
                        XLAT_U:FieldType = 104 ;
                        XLAT_U:MemoryOrder = XY  ;
                        XLAT_U:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT_U:stagger = X ;
                        XLAT_U:units = degree_north ;
                float32 XLAT_V(south_north_stag, west_east) ;
                        XLAT_V:FieldType = 104 ;
                        XLAT_V:MemoryOrder = XY  ;
                        XLAT_V:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT_V:stagger = Y ;
                        XLAT_V:units = degree_north ;
                float32 XLONG(south_north, west_east) ;
                        XLONG:FieldType = 104 ;
                        XLONG:MemoryOrder = XY  ;
                        XLONG:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG:stagger =  ;
                        XLONG:units = degree_east ;
                float32 XLONG_U(south_north, west_east_stag) ;
                        XLONG_U:FieldType = 104 ;
                        XLONG_U:MemoryOrder = XY  ;
                        XLONG_U:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG_U:stagger = X ;
                        XLONG_U:units = degree_east ;
                float32 XLONG_V(south_north_stag, west_east) ;
                        XLONG_V:FieldType = 104 ;
                        XLONG_V:MemoryOrder = XY  ;
                        XLONG_V:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG_V:stagger = Y ;
                        XLONG_V:units = degree_east ;
                datetime64[ns] XTIME() ;
                        XTIME:FieldType = 104 ;
                        XTIME:MemoryOrder = 0   ;
                        XTIME:description = minutes since 2022-12-18 13:00:00 ;
                        XTIME:stagger =  ;
                float32 cwb(time, cwb_channel, south_north, west_east) ;    (35064,4,450,450)
                float32 cwb_center(cwb_channel) ;
                float64 cwb_pressure(cwb_channel) ;
                float32 cwb_scale(cwb_channel) ;
                bool cwb_valid(time) ;                                       35064 = 24 * 365 * 4 + 24 * 1, 0~26303为训练集, 26304~35063为验证集
                <U26 cwb_variable(cwb_channel) ;
                float32 era5(time, era5_channel, south_north, west_east) ;  (35064,20,450,450)
                float32 era5_center(era5_channel) ;
                float64 era5_pressure(era5_channel) ;
                float32 era5_scale(era5_channel) ;
                bool era5_valid(time, era5_channel) ;
                <U19 era5_variable(era5_channel) ;
                datetime64[ns] time(time) ;

    // global attributes:
    }
    """

    path: str

    def __init__(
        self,
        dataset,
        in_channels=(0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19),
        out_channels=(0, 17, 18, 19),
        img_shape_x=448,
        img_shape_y=448,
        roll=False,
        add_grid=True,
        ds_factor=1,
        train=True,
        all_times=False,
        n_history=0,
        min_path=None,
        max_path=None,
        global_means_path=None,
        global_stds_path=None,
        normalization="v1",
    ):
        if not all_times:   # 根据 train 和 all_times 的值，选择适当的时间过滤器来处理数据集
            self._dataset = (
                FilterTime(dataset, is_not_2021)    # 训练集2018~2020
                if train
                else FilterTime(dataset, is_2021)   # 验证集2021
            )
        else:
            self._dataset = dataset

        self.train = train
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.roll = roll
        self.grid = add_grid
        self.ds_factor = ds_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_history = n_history
        self.min_path = min_path
        self.max_path = max_path
        self.global_means_path = (
            to_absolute_path(global_means_path)
            if (global_means_path is not None)
            else None
        )
        self.global_stds_path = (
            to_absolute_path(global_stds_path)
            if (global_stds_path is not None)
            else None
        )
        self.normalization = normalization

    def info(self):
        """Check if the given time is not in the year 2021."""
        return self._dataset.info()

    def __getitem__(self, idx):
        (target, input, _) = self._dataset[idx]     # target.shape=(4,450,450), input.shape=(20,450,450)
        # crop and downsamples 处理数据（如裁剪、降采样）
        # rolling 旋转
        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0

        # channels
        input = input[self.in_channels, :, :]                    # (12,448,448)
        # target = target[self.out_channels, :, :]
        target = target[range(len(self.out_channels)), :, :]     # (4,448,448)

        if self.ds_factor > 1:
            target = self._create_lowres_(target, factor=self.ds_factor)

        reshape_args = (
            y_roll,
            self.train,
            self.n_history,
            self.in_channels,
            self.out_channels,
            self.img_shape_x,
            self.img_shape_y,
            self.min_path,
            self.max_path,
            self.global_means_path,
            self.global_stds_path,
            self.normalization,
            self.roll,
        )
        # SR
        input = reshape_fields(input, "inp", *reshape_args, normalize=False,)
        target = reshape_fields(target, "tar", *reshape_args, normalize=False)

        return target, input, idx

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        in_channels = self._dataset.input_channels()
        # print(in_channels)  # 20个输入变量
        # print(self.in_channels) # [0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19]
        return [in_channels[i] for i in self.in_channels]

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        out_channels = self._dataset.output_channels()
        # print(out_channels) 
        # [ChannelMetadata(name='maximum_radar_reflectivity', level='', auxiliary=False), 
        #   ChannelMetadata(name='temperature_2m', level='', auxiliary=False), 
        #   ChannelMetadata(name='eastward_wind_10m', level='', auxiliary=False), 
        #   ChannelMetadata(name='northward_wind_10m', level='', auxiliary=False)]
        # print(self.out_channels) #[0, 17, 18, 19]
        # return [out_channels[i] for i in self.out_channels]
        return [out_channels[i] for i in range(len(self.out_channels))]

    def __len__(self):
        return len(self._dataset)

    def longitude(self):
        """Get longitude values from the dataset."""
        lon = self._dataset.longitude()
        return lon if self.train else lon[..., : self.img_shape_y, : self.img_shape_x]

    def latitude(self):
        """Get latitude values from the dataset."""
        lat = self._dataset.latitude()
        return lat if self.train else lat[..., : self.img_shape_y, : self.img_shape_x]

    def time(self):
        """Get time values from the dataset."""
        return self._dataset.time()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return (self.img_shape_x, self.img_shape_y)

    def normalize_input(self, x):
        """Convert input from physical units to normalized data. 只有前len(self.in_channels)个变量进行单位标准化,其余变量拼接到标准化后的变量数据"""
        x_norm = self._dataset.normalize_input(x[:, : len(self.in_channels)], channels=self.in_channels)    # 对指定in_channels输入变量标准化
        return np.concatenate((x_norm, x[:, self.in_channels :]), axis=1)   # shape=(len(self.in_channels, self.img_shape_x, self.img_shape_y))

    def denormalize_input(self, x):
        """Convert input from normalized data to physical units."""
        x_denorm = self._dataset.denormalize_input(x[:, : len(self.in_channels)], channels=self.in_channels)
        return np.concatenate((x_denorm, x[:, len(self.in_channels) :]), axis=1)

    def normalize_output(self, x):
        """Convert output from physical units to normalized data."""
        return self._dataset.normalize_output(x, channels=self.out_channels)

    def denormalize_output(self, x):
        """Convert output from normalized data to physical units."""
        return self._dataset.denormalize_output(x, channels=self.out_channels)
    
    # 将图像从低分辨率升采样到高分辨率，用OpenCV的插值方法
    def _create_highres_(self, x, shape):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x

    # 将图像从高分辨率降采样到低分辨率
    def _create_lowres_(self, x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x


def get_zarr_dataset(*, data_path, normalization="v1", all_times=False, **kwargs):
    """Get a Zarr dataset for training or evaluation."""
    data_path = to_absolute_path(data_path)
    get_target_normalization = {"v1": get_target_normalizations_v1, "v2": get_target_normalizations_v2,}[normalization]
    logger.info(f"Normalization: {normalization}")
    zdataset = _ZarrDataset(data_path, get_target_normalization=get_target_normalization)
    return ZarrDataset(dataset=zdataset, normalization=normalization, all_times=all_times, **kwargs)    # 返回数据集
