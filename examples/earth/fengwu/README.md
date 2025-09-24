# FengWu

**模型简介**

如何提高天气预报的时效和准确度，一直是业内的重点课题。随着近年来全球气候变化加剧，极端天气频发，各界对天气预报的时效和精度的期待更是与日俱增。2023年4月7日，上海人工智能实验室联合中国科学技术大学、上海交通大学、南京信息工程大学、中国科学院大气物理研究所及上海中心气象台发布全球中期天气预报大模型“风乌”。基于多模态和多任务深度学习方法构建，AI大模型“风乌”首次实现在高分辨率上对核心大气变量进行超过10天的有效预报，并在80%的评估指标上超越模型GraphCast[1]。此外，“风乌”仅需30秒即可生成未来10天全球高精度预报结果，在效率上大幅优于传统模型。

**模型结构**

“风乌”采用多模态神经网络和多任务自动均衡权重解决多种大气变量表征和相互影响的问题。其针对的大气变量包括：位势、湿度、纬向风速、经向风速、温度以及地表等。“风乌”将这些大气变量看作多模态信息，使用多模态网络结构可以更好地处理这些信息。

研究团队从多任务问题的角度出发，自动学习每个大气变量的重要性，使得多个大气变量之间能够更好地协同优化。为了优化“风乌”的多步预测结果，研究团队提出了“缓存回放”（replay buffer）策略，减少自回归预测误差，提高长期预测的性能。

**数据集准备**

曙光新一代机器平台数据集统一存放在 =  /public/onestore/onedatasets/ERA5

天津体验区统一存放在 = /work/home/onescience2025/osdatasets/FengWu/

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

SHAPE, CHUNKS, DTYPE = (30, 189, 721, 1440), (1, 189, 721, 1440), "float32"
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
    arr = np.random.randn(1, 189, 1, 1).astype(np.float32)
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

## 许可证

FengWu项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。