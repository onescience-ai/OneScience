import h5py
import numpy as np
from ase import Atoms
from ase.io import write
from tqdm import tqdm

def convert_ani1ccx_final(h5_path, output_xyz_path):
    data = h5py.File(h5_path, 'r')
    all_atoms = []
    
    # 单位转换常量 (Hartree -> eV)
    hartree_to_ev = 27.211386245988
    
    # 根据你提供的 Key 列表进行锁定
    energy_key = 'ccsd(t)_cbs.energy'  # 包含括号的准确名称
    forces_key = 'wb97x_dz.forces'      # 使用 DFT 的力
    
    print(f"开始提取数据...")
    print(f"能量来源: {energy_key}")
    print(f"力来源: {forces_key}")

    for mol_id in tqdm(data.keys(), desc="处理分子"):
        group = data[mol_id]
        
        # 确保该分子包含 CC 能量
        if energy_key not in group:
            continue
            
        atomic_numbers = group['atomic_numbers'][:]
        coordinates = group['coordinates'][:]
        cc_energies = group[energy_key][:]
        dft_forces = group[forces_key][:]
        
        # 遍历该分子的所有构型
        for i in range(len(cc_energies)):
            # 过滤掉 NaN（CC计算有时会失败）
            if np.isnan(cc_energies[i]):
                continue
                
            # 创建 ASE 原子对象
            atoms = Atoms(numbers=atomic_numbers, positions=coordinates[i])
            
            # 写入 info 和 arrays (Key 名需匹配你的 MACE 命令)
            atoms.info['DFT_energy'] = cc_energies[i] * hartree_to_ev
            atoms.arrays['DFT_forces'] = dft_forces[i] * hartree_to_ev
            
            all_atoms.append(atoms)

    data.close()

    if len(all_atoms) > 0:
        print(f"成功提取 {len(all_atoms)} 帧有效数据。")
        print(f"正在保存至 {output_xyz_path} ...")
        write(output_xyz_path, all_atoms, format='extxyz')
        print("保存完成！")
    else:
        print("未能提取到有效数据，请检查文件内容。")

if __name__ == "__main__":
    convert_ani1ccx_final('a/public/onestore/onedatasets/MaterialsChemistry/examples/ani1x/ni1x-release.h5', 'a/public/onestore/onedatasets/MaterialsChemistry/examples/ani1x/ni1x_cc_dft.xyz')

