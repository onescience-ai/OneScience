from ase.build import molecule
from ase.io import write
from ase.optimize import LBFGS

from onescience.models.UMA import FAIRChemCalculator
from onescience.models.UMA.units.mlip_unit import load_predict_unit

# ===== 路径：改成你的本地检查点 =====
CKPT = "../checkpoint/uma-s-1p1.pt"  # 替换为你的检查点路径"

# ===== 选项：是否在各自自旋态下弛豫几何（绝热差） =====
RELAX = False
FMAX = 0.05
STEPS = 200

# ===== 载入预测器（GPU）与构造计算器 =====
predictor = load_predict_unit(CKPT, device="cuda")


def make_state(mol_name: str, mult: int, charge: int = 0):
    atoms = molecule(mol_name)
    atoms.pbc = False
    # 关键：同时写入两个键，避免读取不到
    atoms.info["spin"] = mult
    atoms.info["spin_multiplicity"] = mult
    atoms.info["charge"] = charge
    atoms.calc = FAIRChemCalculator(
        predictor, task_name="omol")
    return atoms


# 单重态 / 三重态 CH2
singlet = make_state("CH2_s1A1d", mult=1, charge=0)
triplet = make_state("CH2_s3B1d", mult=3, charge=0)

# （可选）各自自旋面上弛豫
if RELAX:
    for tag, atoms in [("singlet", singlet), ("triplet", triplet)]:
        opt = LBFGS(atoms, logfile=f"{tag}_opt.log")
        opt.run(fmax=FMAX, steps=STEPS)
        write(f"{tag}_final.xyz", atoms)

# 计算单点能（或弛豫后能量）
E_s = singlet.get_potential_energy()
E_t = triplet.get_potential_energy()

gap_ev = E_t - E_s
gap_kcal = gap_ev * 23.060543

print(f"Singlet energy (CH2 1A1):  {E_s:.6f} eV")
print(f"Triplet energy (CH2 3B1):  {E_t:.6f} eV")
print(
    f"Gap (T - S):               {gap_ev:.6f} eV  ({gap_kcal:.2f} kcal/mol)")

# 保存结构，便于复查
write("CH2_singlet.xyz", singlet)
write("CH2_triplet.xyz", triplet)

# 打印自旋/电荷确认，确保不会再触发“未设置自旋”的警告
print(
    "Singlet info:",
    {k: singlet.info[k] for k in (
        "spin", "spin_multiplicity", "charge")},
)
print(
    "Triplet info:",
    {k: triplet.info[k] for k in (
        "spin", "spin_multiplicity", "charge")},
)
