import h5py
data = h5py.File('${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/ani1x-release.h5', 'r')
# 获取第一个分子的所有 Key
first_mol = list(data.keys())[0]
print(f"分子 {first_mol} 包含的 Keys 有:")
print(list(data[first_mol].keys()))
data.close()

