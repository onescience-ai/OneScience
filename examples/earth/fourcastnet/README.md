# FourCastNet

**模型简介**

FourCastNet 是傅立叶预报神经网络的简称，它是一个全球数据驱动天气预报模型，能以 0.25∘ 的分辨率提供准确的全球中短程预报。FourCastNet 可准确预报高分辨率、快速时间尺度的变量，如地表风速、降水和大气水汽。它对规划风能资源、预测热带气旋、热带气旋和大气河流等极端天气事件具有重要意义。对于大尺度变量，FourCastNet 可在短时间内达到最先进的数值天气预报（NWP）模式 ECMWF 综合预报系统（IFS）的预报精度，而对于包括降水在内的具有复杂精细尺度结构的变量，FourCastNet 的预报精度则优于 ECMWF 综合预报系统。FourCastNet 在不到 2 秒的时间内就能生成一周的预报，比 IFS 快了几个数量级。

**模型结构**

FourCastNet使用AFNO模型。该模型网络体系结构是为高分辨率输入而设计的，以ViT为骨干网，并结合了李宗义等人提出的傅里叶神经算子(FNO)。该模型学习函数空间之间的映射，从而求解一系列非线性偏微分方程。 AFNO模型的独创性在于，它将空间混合操作转换为傅里叶变换，混合不同令牌的信息，将特征从空域转换为频域，并对频域特征应用全局可学习滤波器。空间混合复杂度有效地降低到O(NlogN)，其中N是token的数量。

**数据集准备**

曙光新一代机器平台数据集统一存放在 =  /public/onestore/onedatasets/ERA5

天津体验区统一存放在 = /work/home/onescience2025/osdatasets/FourcastNet/

用户如自备数据，则需在conf/config.yaml中指定数据路径，下述路径具体包含内容为：

```
stats_dir: #均值、标准差

checkpoint_dir: #模型文件存储路径
train_data_dir: #训练集
val_data_dir: #验证集
test_data_dir: #推理集
```

在训练阶段数据集目录内需确保存在stats、train、val文件夹；

在推理阶段数据集目录内需确保存在stats、test文件夹；

## 训练

用户如需指定可用卡号，需在终端内根据下述命令指定可用卡号(以使用0号卡和2、3号卡为例，展示2个示例命令)，随后再通过sh脚本进行单机单卡、单机多卡以及多机多卡训练；

```
export HIP_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=2,3
```

单机单卡训练：

```
bash train_single_node_single_device.sh
```

单机多卡训练(默认4卡，可将--nproc_per_node=4中的4改为需要卡数即可)：

```
bash train_single_node_multi_device.sh
```

多机多卡训练：

```
sbatch train_via_slurm.sh
```

运行多机多卡训练前，需确保目录内有logs文件(默认没有该文件夹)

默认每个节点有4卡；

通过修改#SBATCH -N 后面的数字指定节点数，下面以8个节点为例，每个节点4卡，共32卡；

## 推理

单机单卡推理，推理结果将存放在通目录result/下：

```
bash infer_single_node_single_device.sh
```

误差计算，计算所有通道的RMSE及平均RMSE，并给出3个样本3个通道的结果图可视化：

```
python result.py
```

## 模型快速部署测试方法

本节提供随机数据生成，便于用户快速部署模型，进行训练-推理测试。

```
import numpy as np
import h5py
import os
import sys
from onescience.utils.fcn.YParams import YParams

SHAPE, CHUNKS, DTYPE = (30, 20, 721, 1440), (1, 1, 721, 1440), "float32"
# 20可以自行设置，该数字要与config中channels的数量对应即可
def create_h5_files():
    for i in range(2):
        filename = f"{cfg.train_data_dir}/{2000+i}.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("fields", SHAPE, DTYPE, chunks=CHUNKS, data=np.random.randn(*SHAPE).astype(DTYPE))
        print(f"生成文件: {filename}")

    filename = f"{cfg.val_data_dir}/2003.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("fields", SHAPE, DTYPE, chunks=CHUNKS, data=np.random.randn(*SHAPE).astype(DTYPE))
    print(f"生成文件: {filename}")

    filename = f"{cfg.test_data_dir}/2004.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("fields", SHAPE, DTYPE, chunks=CHUNKS, data=np.random.randn(*SHAPE).astype(DTYPE))
    print(f"生成文件: {filename}")


def get_stats():
    # 生成随机数组，第2个维度的20要与上面的第2个维度保持一致即可
    arr = np.random.randn(1, 20, 1, 1).astype(np.float32)
    # 保存数据
    np.save(f'{cfg.stats_dir}/global_stds.npy', arr)
    np.save(f'{cfg.stats_dir}/global_means.npy', arr)

    print(f"已保存到 stats 目录,shape: {arr.shape}, dtype: {arr.dtype}")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    # fourcastnet
    cfg = YParams(config_file_path, 'fourcastnet')

    create_h5_files()

    get_stats()
```

**模型保存：**

修改conf/config.yaml文件中max_epoch为1，该方法用于快速测试训练流程及权重保存方法，便于后续推理测试。

``` 
max_epoch: 1 
```

单机单卡训练一轮，此步骤只为保存模型文件；

```
bash train_single_node_single_device.sh
```

**推理测试：**

本方法除测试推理代码外，还提供耗时统计功能。

单机单卡推理，推理结果将存放在通目录result/下；

在inference.py中对代码做如下修改以统计推理耗时，其中包括总耗时和DCU计算耗时；

```
import time                      # 1.添加代码
start_time =time.perf_counter()  # 1.添加代码

import torch
import os

... ...

fourcastnet_model.eval()

DCU_TIME = 0 # 2.添加代码


with torch.no_grad():
    for j, data in enumerate(test_dataloader):
        invar = data[0].to("cuda:0", dtype=torch.float32)
        outvar = data[1].to("cuda:0", dtype=torch.float32)
        invar = invar[:, :, :-1, :]
        outvar = outvar[:, :, :-1, :]

        st = time.perf_counter()             # 3.添加代码
        outvar_pred = fourcastnet_model(invar)
        DCU_TIME += time.perf_counter()- st  # 4.添加代码
        
        print(f'infer process: {j+1}/{len(test_dataloader)}')
        # pred.append(outvar_pred.cpu().numpy())  # 5.注释代码
        # label.append(outvar.cpu().numpy())      # 5.注释代码


print(f'total DCU time cost {DCU_TIME : .4f}')        # 6.添加代码
end_time = time.perf_counter()                        # 6.添加代码
print(f'total time cost {end_time - start_time:.4f}') # 6.添加代码

# 7.注释代码     
# pred = np.concatenate(pred, axis=0)
# label = np.concatenate(label, axis=0)
# print(pred.shape, label.shape)
# os.makedirs('result/', exist_ok=True)
# np.save("result/pred", pred)
# np.save("result/label", label)
```

运行推理脚本

```
bash infer_single_node_single_device.sh
```

## **在超算互联网使用**

商品地址： https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&id=1872488521565286401&resource=MODEL

## 许可证

FourCastNet项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。