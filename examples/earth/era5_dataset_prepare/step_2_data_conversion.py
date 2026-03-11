import xarray as xr
import h5py
import os
import numpy as np
import glob
import json
import os
from datetime import datetime, timedelta


def generate_datetimes(year, count):
    base = datetime(year, 1, 1, 0)
    return [base + timedelta(hours=6 * i) for i in range(count)]


def save_variable_to_h5(var_name: str, data: np.ndarray, h5_filename: str):

    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('fields', data=data)
    print(f"✅ Saved {h5_filename}")


def process_land_file(nc_path: str, output_dir: str = "."):
    land_var_map = {
        'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        't2m': '2m_temperature',
        'msl': 'mean_sea_level_pressure',
        'sp': 'surface_pressure',
        'tcwv': 'total_column_water_vapour',
        'tp': 'total_precipitation',
        'sst': 'sea_surface_temperature'
    }
    
    ds = xr.open_dataset(nc_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var_name in ds.data_vars:
        var = ds[var_name]
        # 判断是否包含 pressure_level
        filename = f"{land_var_map[var_name]}.h5"
        save_path = os.path.join(output_dir, filename)

        if os.path.exists(save_path):
            print(f'{save_path} exists, skipping...')
        else:
            print(save_path)
            save_variable_to_h5(var_name, var.values, save_path)


def process_pressure_file(nc_path: str, output_dir: str = "."):
    pressure_var_map = {
        'z': 'geopotential',
        'r': 'relative_humidity',
        'q': 'specific_humidity',
        't': 'temperature',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'w': 'vertical_velocity'
    }
    ds = xr.open_dataset(nc_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var_name in ds.data_vars:
        var = ds[var_name]
        # 判断是否包含 pressure_level
        pressure_dim = 'pressure_level'
        levels = ds[pressure_dim].values
        for i, level in enumerate(levels):
            sliced = var.sel({pressure_dim: level})
            filename = f"{pressure_var_map[var_name]}_{int(level)}.h5"
            save_path = os.path.join(output_dir, filename)
            if os.path.exists(save_path):
                print(f'{save_path} exists, skipping...')
            else:
                save_variable_to_h5(var_name, sliced.values, save_path)


# 示例调用
if __name__ == "__main__":
    base_path = "./"
    metadata_path = "./metadata.json"
    new_var_nc = "./nc"      
    output_dir = "./h5"   
    var_map = {
            'u10': '10m_u_component_of_wind',
            'v10': '10m_v_component_of_wind',
            't2m': '2m_temperature',
            'msl': 'mean_sea_level_pressure',
            'sp': 'surface_pressure',
            'tcwv': 'total_column_water_vapour',
            'tp': 'total_precipitation',
            'sst': 'sea_surface_temperature',
            'z': 'geopotential',
            'r': 'relative_humidity',
            'q': 'specific_humidity',
            't': 'temperature',
            'u': 'u_component_of_wind',
            'v': 'v_component_of_wind',
            'w': 'vertical_velocity'
        }

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    exist_years = metadata['years']
    variables = metadata['variables']

    pressure_var_list = ['geopotential']
    # pressure_var_list = ['geopotential', 
    #                     'relative_humidity', 
    #                     'specific_humidity', 
    #                     'temperature', 
    #                     'u_component_of_wind', 
    #                     'v_component_of_wind',
    #                     'vertical_velocity']
    for var in pressure_var_list:
        for year in exist_years:
            output_folder = f"{base_path}/tmp_h5/{year}"
            for nc_file in glob.glob(f'{new_var_nc}/{year}/{var}*'):
                process_pressure_file(nc_file, output_folder)
    
    land_var_list = ['total_precipitation']
    # land_var_list = ['10m_u_component_of_wind', 
    #                 '10m_v_component_of_wind', 
    #                 '2m_temperature', 
    #                 'mean_sea_level_pressure', 
    #                 'surface_pressure', 
    #                 'total_column_water_vapour',
    #                 'total_precipitation',
    #                 'sea_surface_temperature']
    for var in land_var_list:
        for year in exist_years:
            output_folder = f"{base_path}/tmp_h5/{year}"
            nc_file = f'{new_var_nc}/{year}/{var}.nc' 
            process_land_file(nc_file, output_folder)