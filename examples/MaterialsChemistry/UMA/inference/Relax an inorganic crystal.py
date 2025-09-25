import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import LBFGS

from onescience.models.UMA import FAIRChemCalculator
from onescience.models.UMA.units.mlip_unit import load_predict_unit

# === 本地检查点（改成你的实际路径） ===
ckpt = "../checkpoint/uma-s-1p1.pt"

# === 加载预测器与计算器 ===
predictor = load_predict_unit(ckpt, device="cuda")  # GPU 加速
calc = FAIRChemCalculator(predictor, task_name="omat")

# === 建 Fe 体相并绑定计算器 ===
atoms = bulk("Fe")
atoms.calc = calc

# === 原子+晶胞联合弛豫 ===
filt = FrechetCellFilter(atoms)  # 同时优化原子位置与晶胞
opt = LBFGS(filt, logfile="relax.log")  # 记录优化日志到文件
opt.run(fmax=0.05, steps=100)

# === 计算与打印结果 ===
E = atoms.get_potential_energy()
F = atoms.get_forces()
fmax = np.sqrt((F**2).sum(axis=1)).max()
cell = atoms.get_cell()
a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
vol = atoms.get_volume()

print(f"Final energy: {E:.6f} eV  (per atom: {E/len(atoms):.6f} eV/atom)")
print(f"Final fmax: {fmax:.6f} eV/Å")
print(f"Volume: {vol:.6f} Å^3")
try:
    stress = atoms.get_stress()  # Voigt, eV/Å^3
    print("Final stress (Voigt, eV/Å^3):", stress)
except Exception as e:
    print("Stress not available:", e)

print("Cell (Å):\n", cell)
print(
    f"Lengths/angles: a={a:.4f} b={b:.4f} c={c:.4f}  "
    f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}"
)
print("Positions (Å):\n", atoms.get_positions())

# === 保存结果到文件 ===
write("Fe_optimized.cif", atoms)  # 方便后续可视化/复现实验
write("Fe_optimized.traj", atoms)  # ASE 轨迹
np.savetxt("Fe_forces.txt", F)  # 保存力矩阵
