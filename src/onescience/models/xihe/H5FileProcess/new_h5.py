import os
import h5py
import json
import numpy as np
import datetime

INPUT_ROOT = "/public/onestore/onedatasets/ERA5"           # 原始变量.h5目录（每年一个）
OUTPUT_ROOT = "/public/onestore/onedatasets/ERA5/newh5"    # 输出路径
DATA_OUT_DIR = os.path.join(OUTPUT_ROOT, "data")
META_PATH = os.path.join(OUTPUT_ROOT, "metadata.json")

def get_time_list(year):
    """生成固定 1460 个时间戳 (每 6 小时一次)"""
    start = datetime.datetime(year, 1, 1, 0)
    return [(start + datetime.timedelta(hours=6*i)).strftime("%Y%m%d%H") for i in range(1460)]

def scan_variable_order(years):
    """扫描所有年份的变量列表，排序后统一为标准变量顺序"""
    var_sets = {}
    for year in years:
        input_dir = os.path.join(INPUT_ROOT, str(year))
        var_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
        var_names = sorted([f.replace(".h5", "") for f in var_files])
        var_sets[str(year)] = var_names

    # 获取交集 + 顺序：取所有年份变量名交集，按第一个年份顺序排列
    all_sets = [set(v) for v in var_sets.values()]
    var_intersection = set.intersection(*all_sets)

    ref_year = str(years[0])
    ref_order = var_sets[ref_year]
    common_vars = [v for v in ref_order if v in var_intersection]

    print(f"[INFO] Total {len(common_vars)} common variables found across all years.")
    for y, vars_y in var_sets.items():
        diff = set(vars_y) - var_intersection
        if diff:
            print(f"[WARN] Year {y} has extra or unmatched variables: {sorted(diff)}")

    return common_vars

def reorganize_year(year, variables):
    """将原始变量文件合并为每个时间点一个文件，写入 fields"""
    input_dir = os.path.join(INPUT_ROOT, str(year))
    output_dir = os.path.join(DATA_OUT_DIR, str(year))
    os.makedirs(output_dir, exist_ok=True)

    # 打开所有变量文件并收集 dataset handler
    datasets = {}
    for var in variables:
        fpath = os.path.join(input_dir, f"{var}.h5")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing variable: {fpath}")
        datasets[var] = h5py.File(fpath, "r")["fields"]

    times = get_time_list(year)
    for i, ts in enumerate(times):
        out_path = os.path.join(output_dir, f"{ts}.h5")
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} already exists.")
            continue

        with h5py.File(out_path, "w") as out_f:
            arr = np.stack([datasets[v][i, :, :] for v in variables], axis=0)  # [N, 721, 1440]
            out_f.create_dataset("fields", data=arr)
            print(f"[SAVE] {out_path}")

    for ds in datasets.values():
        ds.file.close()

    return {
        "variables": variables,
        "total_files": len(times)
    }

def main():
    years = list(range(2001, 2010))

    print("[STEP 1] Scan and unify variable order ...")
    var_list = scan_variable_order(years)

    print("\n[STEP 2] Reorganize per year ...")
    for year in years:
        print(f"[INFO] Processing year {year} ...")
        reorganize_year(year, var_list)

    # ✅ 新的 metadata 格式
    metadata = {
        "years": [str(y) for y in years],
        "variables": var_list,
        "total_files": 1460
    }

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[DONE] Unified metadata saved to {META_PATH}")
    print(f"[INFO] {len(years)} years, {len(var_list)} variables, {metadata['total_files']} time steps per year.")

if __name__ == "__main__":
    main()