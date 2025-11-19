from ase.build import bulk
from onescience.models.UMA.units.mlip_unit import load_predict_unit
from onescience.datapipes.materials.custom_stack.core.atomic_data import AtomicData, atomicdata_list_to_batch

# 构建多个结构，可替换为 molecule() 或 slab(...)
atoms_list = [
    bulk("Pt"),
    bulk("Cu"),
    bulk("NaCl", crystalstructure="rocksalt", a=2.0),
]

# 转换为 AtomicData 并赋予任务名
atomic_data_list = [
    AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
]

# 合并成一个 batch
batch = atomicdata_list_to_batch(atomic_data_list)

# 加载模型
predictor = load_predict_unit(
    "/public/home/onescience2025404/yuxiaodong/onescience_new/examples/MaterialsChemistry/UMA/uma_finetune_runs/202511-1115-2736-02e4/checkpoints/final/inference_ckpt.pt",#替换为你的检查点路径",#替换为你的检查点路径
    device="cuda"
)

# 执行推理
preds = predictor.predict(batch)

# 输出每个结构的能量和原子力
for i, atoms in enumerate(atoms_list):
    energy = preds["energy"][i].item()
    forces = preds["forces"][batch.batch == i].cpu().numpy()

    print(f"\nStructure #{i + 1}: {atoms.get_chemical_formula()}")
    print("Predicted energy:", energy)
    print("Predicted forces:\n", forces)
