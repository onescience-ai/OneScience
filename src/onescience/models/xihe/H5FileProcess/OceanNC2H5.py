import xarray as xr
import h5py
import os
import numpy as np
import sys
import json
import os
from datetime import datetime, timedelta
from scipy.ndimage import zoom
# import matplotlib.pyplot as plt
import calendar


var_map = {'uo': 'eastward_sea_water_velocity',
           'vo': 'northward_sea_water_velocity',
           'so': 'sea_water_salinity',
           'thetao': 'sea_water_potential_temperature',
           'analysed_sst': 'sea_surface_temperature',
           'zos': 'sea_surface_height_above_geoid'
           } 
#海洋数据

#数据插值：线性插值
def interpolate(data, target_shape=[2041, 4320]):
    original_shape = data.shape
    zoom_factors = tuple(target_size / original_size for target_size, original_size in zip(target_shape, original_shape))
    interpolated_data = zoom(data, zoom_factors, order=1)  # `order=1` 表示线性插值
    return interpolated_data


def process_file(nc_path, output_dir, filename):
    
    ds = xr.open_dataset(nc_path)

    for var_name in ds.data_vars:
        print(f'processing {var_map[var_name]}...', end='')
        
        save_path = f'{output_dir}/{filename}'
        if os.path.exists(save_path):
            print(f'{save_path} exists, skipping...')
        else:
            data = ds[var_name].values.squeeze()
            if data.shape != (2041, 4320):
                print(f'data shape: {data.shape}, interpolate to (2041, 4320), ', end='')
                data = interpolate(data, target_shape=[2041, 4320])
            with h5py.File(save_path, 'w') as h5f:
                h5f.create_dataset('fields', data=data)
            print(f'{filename} saving done.')


def get_uv_from_ERA5(year, output_dir):
    metadata_path = "/public/onestore/onedatasets/ERA5/newh5/metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    years = metadata['years']
    variables = metadata['variables']
    if year not in years:
        print(f'❌ ❌ {years} 10m_u and 10m_v of ERA5 are not provided now, contact to data manager...')
        exit()
    u_idx = variables.index('10m_u_component_of_wind') #在通道中的下标
    v_idx = variables.index('10m_v_component_of_wind')

    stop = False   # 用来标记是否跳出全部循环
    flag=0
    month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    for month in month_list:
        if stop:
            break  # flag 达到 365 时跳出外层循环
        # 每个月日期不同，该方法可以获取当月日期
        _, num_days = calendar.monthrange(int(year), int(month))
        day_list = [f"{day:02}" for day in range(1, num_days + 1)]
        for day in day_list:
            with h5py.File(f'/public/onestore/onedatasets/ERA5/newh5/data/{year}/{year}{month}{day}00.h5', 'r') as f: #era5数据集508位置
                u10 = f['fields'][u_idx, :, :]  # 第 i 帧放入第 j 通道
                if u10.shape != (2041, 4320):
                    print(f'u10 shape: {u10.shape}, interpolate to (2041, 4320), ', end='')
                    u10 = interpolate(u10, target_shape=[2041, 4320])
                    save_path = f'{output_dir}/{year}{month}{day}_10m_u_component_of_wind.h5'
                    if os.path.exists(save_path):
                        print(f'{save_path} exists, skipping...')
                    else:
                        with h5py.File(save_path, 'w') as h5f:
                            h5f.create_dataset('fields', data=u10)
                        print(f'{year}{month}{day}_10m_u_component_of_wind.h5 saving done.')
                        
            with h5py.File(f'/public/onestore/onedatasets/ERA5/newh5/data/{year}/{year}{month}{day}00.h5', 'r') as f: #era5
                v10 = f['fields'][v_idx, :, :]  # 第 i 帧放入第 j 通道
                if v10.shape != (2041, 4320):
                    print(f'v10 shape: {v10.shape}, interpolate to (2041, 4320), ', end='')
                    v10 = interpolate(v10, target_shape=[2041, 4320])
                    save_path = f'{output_dir}/{year}{month}{day}_10m_v_component_of_wind.h5'
                    if os.path.exists(save_path):
                        print(f'{save_path} exists, skipping...')
                    else:
                        with h5py.File(save_path, 'w') as h5f:
                            h5f.create_dataset('fields', data=v10)
                        print(f'{year}{month}{day}_10m_v_component_of_wind.h5 saving done.')
            flag += 1
            if flag >= 365:
                stop = True   # 设置跳出标志
                break          # 先跳出 day 这一层循环


# 示例调用
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_means.py <your_h5_file.h5>")
        sys.exit(1)
    
    year = sys.argv[1]

    base_path = "/public/onestore/onedatasets/CMEMS/newdata/"
    # base_path = "/public/home/onescience2025404/hanym/data/CMEMS/newdata/"
    output_folder = f"{base_path}/tmp_h5/{year}"      
    os.makedirs(output_folder, exist_ok=True)
    month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    depth_list = list(range(1, 24))
    
    #获取 u10 v10
    get_uv_from_ERA5(str(year), f"{base_path}/tmp_h5/{year}/")

    flag_day = 0
    stop = False
    # 获取 sst 来自osita，高分辨率
    for month in month_list:
        if stop:
            break  # flag 达到 365 时跳出外层循环
        # 每个月日期不同，该方法可以获取当月日期
        _, num_days = calendar.monthrange(int(year), int(month))
        day_list = [f"{day:02}" for day in range(1, num_days + 1)]
        for day in day_list:
            nc_file = f"{base_path}/nc/sst/{year}/{month}/{year}{month}{day}_sst.nc"         #      # 
            process_file(nc_file, f"{base_path}/tmp_h5/{year}/", f'{year}{month}{day}_sea_surface_temperature.h5')
            flag_day += 1
            if flag_day >= 365:
                stop = True   # 设置跳出标志
                print(f'{year}_sea_surface_temperature  saving done')
                break          # 先跳出 day 这一层循环
    #获取ssh
    flag_day = 0
    stop = False
    for month in month_list:
        if stop:
            break  # flag 达到 365 时跳出外层循环
        # 每个月日期不同，该方法可以获取当月日期
        _, num_days = calendar.monthrange(int(year), int(month))
        day_list = [f"{day:02}" for day in range(1, num_days + 1)]
        for day in day_list:
            nc_file = f"{base_path}/nc/ssh/{year}/{month}/{year}{month}{day}_ssh.nc"         #      # 
            process_file(nc_file, f"{base_path}/tmp_h5/{year}/", f'{year}{month}{day}_sea_surface_height_above_geoid.h5')
            
            flag_day += 1
            if flag_day >= 365:
                stop = True   # 设置跳出标志
                print(f'{year}_sea_surface_height_above_geoid  saving done')
                break          # 先跳出 day 这一层循环            
   #23*4
    pressure_list = ['uo','vo','so','thetao'] #'uo','vo','so','thetao'
    for pressure in pressure_list:
        flag_day = 0
        stop = False
        for month in month_list:
            if stop:
                break  # flag 达到 365 时跳出外层循环
            # 每个月日期不同，该方法可以获取当月日期
            _, num_days = calendar.monthrange(int(year), int(month))
            day_list = [f"{day:02}" for day in range(1, num_days + 1)] #["01","02","03"]
            for day in day_list:
                for depth in depth_list:
                    nc_file = f"{base_path}/nc/{pressure}/{year}/{month}/{year}{month}{day}_{pressure}_{depth}.nc"         #      # 
                    process_file(nc_file, f"{base_path}/tmp_h5/{year}/", f'{year}{month}{day}_{var_map[pressure]}_{depth}.h5')
                flag_day += 1
                if flag_day >= 365:
                    stop = True   # 设置跳出标志
                    print(f'{year}_{var_map[pressure]}_{depth}    done')
                    break          # 先跳出 day 这一层循环