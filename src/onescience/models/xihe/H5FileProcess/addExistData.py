import xarray as xr
import h5py
import os
import numpy as np
import sys
import json
import os
from datetime import datetime, timedelta

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
        'tp': 'total_precipitation'
    }
    ds = xr.open_dataset(nc_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var_name in ds.data_vars:
        print(f'processing {var_name}...')
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
        'v': 'v_component_of_wind'
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


def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_all_files_exist(variables, base_dir='.'):
    missing = []
    for var in variables:
        if not os.path.exists(os.path.join(base_dir, f"{var}.h5")):
            missing.append(var)
    return missing

def open_h5_handles(variables, input_dir):
    handles = {}
    for var in variables:
        file_path = os.path.join(input_dir, f"{var}.h5")
        if not os.path.exists(file_path):
            # raise FileNotFoundError(f"Missing file: {file_path}")
            print(f"Missing file: {file_path}")
        handles[var] = file_path 
    return handles


def generate_datetimes(year, count):
    base = datetime(year, 1, 1, 0)
    return [base + timedelta(hours=6 * i) for i in range(count)]


def regroup_by_time_streaming(year, variables, handles, output_dir):
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    N = 1460

    M = len(variables)
    datetimes = generate_datetimes(year, N)

    for i in range(N):
        dt_str = datetimes[i].strftime('%Y%m%d%H')
        out_path = os.path.join(year_dir, f"{dt_str}.h5")
        if os.path.exists(out_path):
            print(f'{out_path} exists, skipping...')
            continue
        data = np.empty((M, 721, 1440), dtype=np.float32)

        for j, var in enumerate(variables):
            with h5py.File(handles[var], 'r') as f:
                data[j] = f[list(f.keys())[0]][i]  # 第 i 帧放入第 j 通道

        
        with h5py.File(out_path, 'w') as out_f:
            out_f.create_dataset('fields', data=data)

        print(f"✅ Written {i}/{N}: {dt_str}.h5")


# 示例调用
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_means.py <your_h5_file.h5>")
        sys.exit(1)
    
    year = sys.argv[1]

    base_path = "/public/onestore/onedatasets/ERA5/newh5/newdata/"


    output_folder = f"{base_path}/tmp_h5/{year}"      
    land_list = {'10m_u_component_of_wind',
                 '10m_v_component_of_wind',
                 '2m_temperature',
                 'mean_sea_level_pressure',
                 'surface_pressure',
                 'total_column_water_vapour',
                 'total_precipitation'
    }
    for land in land_list: 
        nc_file = f"{base_path}/nc/{year}/{land}.nc" 
        process_land_file(nc_file, output_folder)

    pressure_list = {'geopotential',
                     'relative_humidity',
                     'specific_humidity',
                     'temperature',
                     'u_component_of_wind',
                     'v_component_of_wind'}
    for pl in pressure_list:
        nc_file = f"{base_path}/nc/{year}/{pl}_pre1.nc"         # 
        process_pressure_file(nc_file, output_folder)
        nc_file = f"{base_path}/nc/{year}/{pl}_pre2.nc"         # 
        process_pressure_file(nc_file, output_folder)
        nc_file = f"{base_path}/nc/{year}/{pl}_pre3.nc"         # 
        process_pressure_file(nc_file, output_folder)
        nc_file = f"{base_path}/nc/{year}/{pl}_pre.nc" 
        process_pressure_file(nc_file, output_folder)

    metadata_path = "/public/onestore/onedatasets/ERA5/newh5/metadata.json"
    input_dir = "/public/onestore/onedatasets/ERA5/newh5/newdata/tmp_h5"         # <-- 替换为你的输入路径
    output_dir = "/public/onestore/onedatasets/ERA5/newh5/newdata/h5"    # <-- 替换为你的输出路径

    metadata = load_metadata(metadata_path)
    # year = [f'{year
    variables = metadata['variables']
    
    handles = open_h5_handles(variables, f'{input_dir}/{year}')
    exit()
    if check_all_files_exist(variables, f'{input_dir}/{year}'):
        print(f"❌ Missing files:")
    else:
        print(f'{year} all files correct, need {len(variables)}, has {len(handles)}')
        print(f"🌀 Processing year {year}")
        regroup_by_time_streaming(int(year), variables, handles, output_dir) 