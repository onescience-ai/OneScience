# path: regroup_h5_by_time_streaming.py

import json
import os
import h5py
import numpy as np
import sys
from datetime import datetime, timedelta
from tqdm import tqdm


def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_folders(path, years):
    # 获取路径下所有的文件夹
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f.isdigit()]
    
    # 将文件夹名称转换为整数，并与 years 进行排序
    folders = sorted([folder for folder in folders])
    years = sorted(years)
    
    # 检查文件夹名称是否包含所有 years 列表中的年份（不要求严格匹配，只要包含）
    missing_years = [year for year in years if year not in folders]
    return missing_years
    


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

def merge(inputname):

    metadata_path = "/public/onestore/onedatasets/ERA5/newh5/new_metadata.json"
    input_dir = "/public/onestore/onedatasets/ERA5/newh5/newdata/tmp_h5"         # <-- 替换为你的输入路径
    output_dir = "/public/onestore/onedatasets/ERA5/newh5/data/newdata"    # <-- 替换为你的输出路径
    final_dir = '/public/onestore/onedatasets/ERA5/newh5/data/newdata2/'
    metadata = load_metadata(metadata_path)
    years = metadata['years']
    variables = metadata['variables']

    missing_years = check_folders(input_dir, years)
    variables.append(inputname[:-3])
    for year in years:
        if missing_years:
            print("❌ Some years are missing from the folders.")
            print(f"Missing years: {missing_years}")
        else:
            print("✅ All years in the list are covered by the folders.")
            print(f"🌀 Processing year {year}")
            year_folder_ori = os.path.join(output_dir, str(year))
            year_folder_new = os.path.join(input_dir, str(year))
            # 获取 new_data_path 下的所有 h5 文件 (形状: [1460, H, W])
            print(f"🌀 Processing {inputname}")
            # 读取 new_data_path 中的 h5 文件 (形状: [1460, H, W])
            
            path2_file = os.path.join(year_folder_new, inputname)
            with h5py.File(path2_file, 'r') as f2:
                data2 = f2['fields'][:]  # 假设数据保存在 'data' 变量中
            # 获取 ori_path 中的对应文件夹并逐个处理文件 (形状: [M, H, W])
            year_folder_ori = os.path.join(output_dir, str(year))
            filelist = sorted(os.listdir(year_folder_ori))
            output_path = f'{final_dir}/{year}/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for i in tqdm(range(len(filelist))):
                filename = filelist[i]
                output_name = os.path.join(output_path, filename)
                if os.path.exists(output_name):
                    print('skipping ', output_name)
                    continue
                # 读取 ori_path 中的 h5 文件
                path1_file = os.path.join(year_folder_ori, filename)
                with h5py.File(path1_file, 'r') as f1:
                    data1 = f1['fields'][:]  # 假设数据保存在 'data' 变量中
                # 拼接数据 (将 [M, H, W] 数据加到 [1460, H, W] 上)
                new_data = np.concatenate((data1, data2[i: i+1]), axis=0)
                # # 更新原文件，或者保存到新的文件中
                with h5py.File(output_name, 'w') as f1:
                    f1.create_dataset('fields', data=new_data)
    metadata_path = "/public/onestore/onedatasets/ERA5/newh5/new_metadata.json"
    metadata['variables'] = variables
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
                    
        
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: input the filename")
        sys.exit(1)
    
    filename = sys.argv[1]
    merge(filename)
## 输入文件名，带着.h5