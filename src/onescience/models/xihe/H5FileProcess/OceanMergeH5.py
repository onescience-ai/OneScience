import xarray as xr
import h5py
import os
import numpy as np
import sys
import json
import os
from datetime import datetime, timedelta
import calendar

pressure_list = ['eastward_sea_water_velocity',
                 'northward_sea_water_velocity',
                 'sea_water_salinity',
                 'sea_water_potential_temperature']

surface_list = ['sea_surface_temperature', 
                'sea_surface_height_above_geoid',
                '10m_u_component_of_wind', 
                '10m_v_component_of_wind']

depth_list = list(range(1, 24))



def merge(year, variables, input_dir, output_dir):
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(output_dir, exist_ok=True)

    M = len(variables)
    month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    flag_day = 0
    stop = False
    for month in month_list:
        if stop:
            break  # flag 达到 365 时跳出外层循环
        _, num_days = calendar.monthrange(int(year), int(month))
        day_list = [f"{day:02}" for day in range(1, num_days + 1)]
        for day in day_list:
            data = np.empty((M, 2041, 4320), dtype=np.float32)
            filename = f'{output_dir}/{year}{month}{day}.h5'
            if os.path.exists(filename):
                print(f'{filename} exists, skipping...')
            else:
                for j, var in enumerate(variables):
                    with h5py.File(f'{input_dir}/{year}{month}{day}_{var}.h5', 'r') as f:
                        data[j] = f['fields'][:]  # 第 i 帧放入第 j 通道

                with h5py.File(filename, 'w') as out_f:
                    out_f.create_dataset('fields', data=data)
                print(f'✅ ✅ {filename} saving done...')
            flag_day += 1
            if flag_day >= 365:
                stop = True   # 设置跳出标志
                break          # 先跳出 day 这一层循环 


def check_all_files_exist(variables, input_dir):
    missing = []
    month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    flag_day = 0
    stop = False
    for month in month_list:
        if stop:
            break  # flag 达到 365 时跳出外层循环
        _, num_days = calendar.monthrange(int(year), int(month))
        day_list = [f"{day:02}" for day in range(1, num_days + 1)]
        for day in day_list:
            for var in variables:
                filename = f'{input_dir}/{input_dir[-4:]}{month}{day}_{var}.h5'
                if not os.path.exists(filename):
                    missing.append(filename)
            flag_day += 1
            if flag_day >= 365:
                stop = True   # 设置跳出标志
                break          # 先跳出 day 这一层循环 
    return missing


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: input year")
        sys.exit(1)
    
    year = sys.argv[1]

    # base_path = "/public/onestore/onedatasets/CMEMS/newdata/"
    

    metadata_path = "/public/onestore/onedatasets/CMEMS/metadata.json"
    input_dir = f"/public/onestore/onedatasets/CMEMS/newdata/tmp_h5/{year}"         # <-- 替换为你的输入路径,转换成h5的变量数据
    output_dir = f"/public/onestore/onedatasets/CMEMS/newdata/h5/{year}"    # <-- 替换为你的输出路径，输出为每一天的数据，每个都包含所有变量

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    variables = metadata['variables']
    missing = check_all_files_exist(variables, input_dir)
    if missing:
        print(f"❌ Missing files: {missing}")
        exit()
    else:
        print(f'{year} all files correct, 🌀 Processing year {year}')
        merge(year, variables, input_dir, output_dir)
        
