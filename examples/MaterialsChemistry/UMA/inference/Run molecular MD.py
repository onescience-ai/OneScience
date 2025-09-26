from ase import units
from ase.build import molecule
from ase.io import Trajectory, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from onescience.models.UMA import FAIRChemCalculator
from onescience.models.UMA.units.mlip_unit import load_predict_unit

# === 本地检查点路径（改成你的实际路径）===
ckpt = "../checkpoint/uma-s-1p1.pt"

# === 加载预测器与计算器 ===
predictor = load_predict_unit(ckpt, device="cuda")  # GPU加速
calc = FAIRChemCalculator(
    predictor, task_name="omol")  # 分子任务

# === 构建体系 ===
atoms = molecule("H2O")  # 可替换为其它分子
atoms.calc = calc

# === 速度初始化（400 K），去线动量/角动量 ===
T0 = 400  # K
MaxwellBoltzmannDistribution(atoms, temperature_K=T0)
Stationary(atoms)  # 去整体平动
ZeroRotation(atoms)  # 去整体转动

# === 动力学设置：Langevin ===
dt = 0.1 * units.fs
gamma = 0.001 / units.fs
dyn = Langevin(atoms, timestep=dt,
               temperature_K=T0, friction=gamma)

# === 轨迹与日志 ===
traj = Trajectory("my_md.traj", "w", atoms)  # ASE 原生轨迹
dyn.attach(traj.write, interval=1)  # 每步写一帧
logger = MDLogger(dyn, atoms, "md.log",
                  header=True, stress=False, peratom=False)
dyn.attach(logger, interval=10)  # 每10步记录一次


# 也打印到屏幕（可选）
def printer():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    Tinst = 2.0 * ekin / (3 * len(atoms) * units.kB)
    print(
        f"Step {dyn.nsteps:5d}  Epot={epot: .6f} eV  Ekin={ekin: .6f} eV  T={Tinst: .1f} K"
    )


dyn.attach(printer, interval=50)

# === 运行MD ===
steps = 1000
dyn.run(steps=steps)

# === 结束后输出关键信息与保存 ===
Epot = atoms.get_potential_energy()
Ekin = atoms.get_kinetic_energy()
Tfinal = 2.0 * Ekin / (3 * len(atoms) * units.kB)
print("\n== MD Finished ==")
print(
    f"Final Epot = {Epot:.6f} eV,  Ekin = {Ekin:.6f} eV,  T = {Tfinal:.2f} K")
print("COM =", atoms.get_center_of_mass())

# 保存最终结构（xyz 与 cif）
write("final.xyz", atoms)
try:
    write("final.cif", atoms)  # 分子一般无PBC，cif可有可无
except Exception as e:
    print("Save CIF skipped:", e)

# 如需导出整段轨迹为 .xyz（多帧），取消注释以下两行
# frames = read("my_md.traj", ":")
# write("my_md_all.xyz", frames)
