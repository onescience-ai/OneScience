# GraphCast 

**模型简介**

Graphcast是谷歌Deepmind提出的一种用于全球中期天气预报的的方法，同时支持包括预测热带气旋路径、大气河流和极端气温在内的应用。该模型将地球天气的两个最新状态(当前时间和六个小时前)作为输入，并预测六小时后的天气状态。一个单独的天气状态由一个0.25维度/经度网格(721x1440)表示，这相当于赤道上大约28x28公里的分辨率，其中每个网格点代表一组地面和大气变量。本项目是利用torch框架对GraphCast的重构版本，主要用于模型训练。

**模型结构** 

Graphcast是一种基于机器学习的天气预报大模型，性能优于世界上最准确的机器学习天气预报系统。Deepmind以编码-处理-解码的方式使用图神经网络(Graph Neural Network，GNN)来创建一个自回归模型。 Graphcast三阶段模拟过程如下： 1）第一阶段：编码阶段基于单个GNN将输入网格上表示为节点属性的变量映射到多网格图表示上的学习节点。 2）第二阶段：处理器阶段使用16个非共享GNN层来在多网格上执行学习信息传递，使得能够以很少的消息传递步骤进行有效的本地和远程信息传递。 3）第三阶段：解码器将来自多网格表示的第二阶段最终学习到的特征映射回纬度-经度网格，与第一阶段类似使用单个GNN层，并将输出预测作为残差更新最新的输入状态。

**数据集准备** 

曙光新一代机器平台数据集统一存放在 =  /public/onestore/onedatasets/ERA5

天津体验区统一存放在 = /work/home/onescience2025/osdatasets/graphcast/

用户如自备数据，则需在conf/config.yaml中指定数据路径，下述路径具体包含内容为：

```
stats_dir: #均值、标准差
static_dataset_path: #地势、海洋陆地掩码等
checkpoint_dir: #模型文件存储路径
train_data_dir: #训练集
val_data_dir: #验证集
test_data_dir: #推理集
```

在训练阶段数据集目录内需确保存在stats、static、train、val文件夹；

在推理阶段数据集目录内需确保存在stats、static、test文件夹；

首先需要运行compute_time_diff_std.py以及get_data_json.py在目录下得到time_diff_mean.npy、time_diff_std.npy以及data.json，命令参考如下：

```
python compute_time_diff_std.py
python get_data_json.py
```

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

默认每个节点有4卡，不同集群可以修改#SBATCH --gres=dcu:4数量、export HIP_VISIBLE_DEVICES=0,1,2,3可见卡数以及torchrun中的--nproc_per_node=4数量；

**微调**

Graphcast训练模式为训练+微调，微调阶段通过修改预测迭代步长实现更准确的长期预测，如需进行微调，必须通过训练得到checkpoints/graphcast.pth权重文件。

单机单卡微调：

```
bash finetune_single_node_single_device.sh
```

单机多卡微调(默认4卡)：

```
bash finetune_single_node_multi_device.sh
```

多机多卡微调：

```
sbatch finetune_via_slurm.sh
```

运行多机多卡微调前，需确保模型经过训练且保存过checkpoints/graphcast.pth，同时，使用slurm系统需确保目录内有logs文件(默认没有该文件夹)

## 推理

单机单卡推理，推理结果将存放在目录result/下，如需进行推理，则需要进行微调得到checkpoints/graphcast_finetune.pth，若不进行微调，可通过修改推理代码中加载的权重文件名称，来获取训练阶段得到的权重进行推理：

```
bash infer_single_node_single_device.sh
```

误差计算，计算所有通道的RMSE及平均RMSE，并给出3个样本3个通道的结果图可视化，可视化结果在result/*.png：

```
python result.py
```

## 模型快速部署测试方法

本节提供随机数据生成，便于用户快速部署模型，进行训练-微调-推理测试。

训练时间较长，若想快速测试模型的训练-微调-推理过程，可在train_graphcast.py以及finetune_graphcast.py的下面部分进行临时断点，请注意该方法仅为快速保存权重用于后续测试，两个文件类似，按照下面定位位置即可。

```
train文件
......
print_length = 1  # also can set it to 'len(train_dataloader) // 64'
epoch_start_time = time.perf_counter()

for epoch in range(cfg.num_iters_step1 + cfg.num_iters_step2):
......
......
        if (i + 1) % cfg.val_freq == 0:
            graphcast_model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                val_batch_time = time.perf_counter()
                for j, data in enumerate(val_dataloader):
                    ##  加入该break语句
                    if j == 1:
                        break
                    invar = data[0].to(device=local_rank)
                    outvar = data[1].to(device=local_rank)
......
......
		valid_loss /= len(val_dataloader)
                    is_save_ckp = False
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_loss_epoch = i
                        world_rank == 0 and save_checkpoint(graphcast_model,
                                                            optimizer,
                                                            scheduler,
                                                            best_valid_loss,
                                                            best_loss_epoch,
                                                            cfg.checkpoint_dir)
                        is_save_ckp = True
                        ## 加入该exit退出语句
                        exit()
                        if world_rank == 0:
                            logger.info(f"Best loss at Minibatch: {i + 1}" + (", saving checkpoint" if is_save_ckp else ""))
......
......
```

数据生成方法如下，

```
import numpy as np
import h5py
import os
import sys
import xarray as xr
from onescience.utils.fcn.YParams import YParams
SHAPE, CHUNKS, DTYPE = (30, 69, 721, 1440), (1, 1, 721, 1440), "float32"
# 69可以自行设置，原始模型为83，该数字要与config中channels的数量对应即可
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
    # 生成随机数组，第2个维度的69要与上面的第2个维度保持一致即可
    arr = np.random.randn(1, 69, 1, 1).astype(np.float32)
    # 保存数据
    np.save(f'{cfg.stats_dir}/global_stds.npy', arr)
    np.save(f'{cfg.stats_dir}/global_means.npy', arr)

    print(f"已保存到 stats 目录,shape: {arr.shape}, dtype: {arr.dtype}")

def get_data(var, name):
    ds = xr.Dataset(
        data_vars={
            f"{var}": (("valid_time", "latitude", "longitude"),
                np.random.rand(1, 721, 1440).astype(np.float32))
        },
        coords={
            "valid_time": ["2015-12-31"],
            "latitude": np.linspace(90, -90, 721, dtype=np.float64),
            "longitude": np.linspace(0, 359.75, 1440, dtype=np.float64),
            "number": 0,
            "expver": "",
        },
        attrs={
            "GRIB_centre": "ecmf",
            "GRIB_centreDescription": "European Centre for Medium-Range Weather Forecasts",
            "GRIB_subCentre": "0",
            "Conventions": "CF-1.7",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "history": "Generated manually",
        }
    )

    ds.to_netcdf(f"{cfg.static_dataset_path}/{name}.nc")



if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    # 在测试其他模型时，graphcast需要修改为config中对应名称
    cfg = YParams(config_file_path, 'graphcast')

    create_h5_files()

    get_stats()

    get_data('z', 'geopotential')
    get_data('lsm', 'land_sea_mask')
```

## 许可证 (格式：H5)

Graphcast项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。