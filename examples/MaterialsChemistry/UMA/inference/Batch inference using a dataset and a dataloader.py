from torch.utils.data import DataLoader
from onescience.datapipes.uma.ase_datasets import AseDBDataset
from onescience.datapipes.uma.atomic_data import atomicdata_list_to_batch
from onescience.utils.UMA.mlip_unit import load_predict_unit

# 数据集路径
db_path = "../dataset/omat24/val/rattled-300-subsampled/data.aselmdb"#替换为你的数据集路径

# 加载数据集
dataset = AseDBDataset(
    config=dict(
        src=db_path,
        a2g_args=dict(task_name="omat")  # 确保任务名称与OMat24数据集匹配
    )
)

# 创建DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=atomicdata_list_to_batch
)

# 加载UMA模型
predictor = load_predict_unit(
    "../checkpoint/uma-s-1p1.pt",#替换为你的检查点路径
    device="cuda"
)

# 批量推理
for i, batch in enumerate(loader):
    preds = predictor.predict(batch)

    for j in range(len(preds["energy"])):
        energy = preds["energy"][j].item()
        forces = preds["forces"][batch.batch == j].cpu().numpy()

        print(f"\n[Batch {i} | Structure {j}]")
        print("Predicted energy:", energy)
        print("Predicted forces:\n", forces)
