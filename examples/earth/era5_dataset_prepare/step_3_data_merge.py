import xarray as xr
import h5py
import os
import numpy as np
import sys
import json
import os
from datetime import datetime, timedelta


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
                data[j] = f[list(f.keys())[0]][i]  
        
        with h5py.File(out_path, 'w') as out_f:
            out_f.create_dataset('fields', data=data)

        print(f"✅ Written {i}/{N}: {dt_str}.h5")


# 示例调用
if __name__ == "__main__":

    metadata_path = "./metadata.json"
    input_dir = "./tmp_h5"     
    output_dir = "./h5"   

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    variables = metadata['variables']
    years = metadata['years']
    for year in years:
        handles = open_h5_handles(variables, f'{input_dir}/{year}')
        if check_all_files_exist(variables, f'{input_dir}/{year}'):
            print(f"❌ Missing files:")
        else:
            print(f'{year} all files correct, need {len(variables)}, has {len(handles)}')
            print(f"🌀 Processing year {year}")
            regroup_by_time_streaming(int(year), variables, handles, output_dir) 