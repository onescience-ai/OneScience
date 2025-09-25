from ase.build import add_adsorbate, bulk, fcc100, molecule

from onescience.datapipes.uma.atomic_data import AtomicData, atomicdata_list_to_batch
from onescience.models.UMA.units.mlip_unit import load_predict_unit

# 1. 创建异构结构
h2o = molecule("H2O")
h2o.info.update({"charge": 0, "spin": 1})

pt = bulk("Pt")

slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

# 2. 结构转为 AtomicData，并指定不同 task_name
atomic_data_list = [
    AtomicData.from_ase(
        h2o, task_name="omol", r_data_keys=["spin", "charge"], molecule_cell_size=12
    ),
    AtomicData.from_ase(pt, task_name="omat"),
    AtomicData.from_ase(slab, task_name="oc20"),
]

# 3. 合成 batch
batch = atomicdata_list_to_batch(atomic_data_list)

# 4. 加载 UMA 模型
predictor = load_predict_unit(
    "../checkpoint/uma-s-1p1.pt", device="cuda"  # 替换为你的检查点路径
)

# 5. 执行联合推理
preds = predictor.predict(batch)

# 6. 输出每个结构结果
for i in range(len(preds["energy"])):
    energy = preds["energy"][i].item()
    forces = preds["forces"][batch.batch == i].cpu().numpy()

    print(f"\n[Structure {i}]")
    print("Predicted energy:", energy)
    print("Predicted forces:\n", forces)
