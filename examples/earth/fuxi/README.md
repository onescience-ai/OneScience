# Fuxi

**模型简介**

Fuxi模型是由复旦大学的研究人员开发的一个基于数据驱动的全球天气预报模型，它摒弃了传统复杂的微分方程，转而通过多阶段机器学习架构，可提供15天的全球预报。时间分辨率为6小时，空间分辨率为0.25°，相当于赤道附近约25公里 x 25公里的范围，使用ECMWF39年的ERA5再分析数据集训练，在15天预报尺度上实现了效率与精度的双重突破。

**模型结构**

基本的伏羲模型体系结构由三个主要组件组成，Cube Embedding、U-Transformer(Swin-Transformer)和全连接层。输入数据结合了上层空气和地表变量，并创建了一个维度为69×720×1440的数据立方体，以一个时间步作为一个step。高维输入数据通过联合时空Cube Embedding进行维度缩减，转换为C×180×360。Cube Embedding的主要目的是减少输入数据的时空维度，减少冗余信息。随后，U-Transformer处理嵌入数据，并使用简单的全连接层进行预测，输出首先被重塑为69×720×1440。

**数据集准备**

曙光新一代机器平台数据集统一存放在 =  /public/onestore/onedatasets/ERA5

天津体验区统一存放在 = /work/home/onescience2025/osdatasets/Fuxi/

用户如自备数据，则需在conf/config.yaml中指定数据路径，下述路径具体包含内容为：

```
stats_dir: #均值、标准差

checkpoint_dir: #模型文件存储路径
train_data_dir: #训练集
val_data_dir: #验证集
test_data_dir: #推理集
```

## 训练

用户如需指定可用卡号，需在终端内根据下述命令指定可用卡号(以使用0号卡和2、3号卡为例，展示2个示例命令)，随后再通过sh脚本进行单机单卡、单机多卡以及多机多卡训练；

```
export HIP_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=2,3
```

fuxi分为4个版本，base、short、medium、以及long，分别对应单步预测、短期(5天)预测、中期(10天)预测、长期(15天)预测。

其中，base版本使用t和t+1时刻预测t+2时刻，short版本则是通过t和t+1预测t+2后，通过t+1和t+2预测值迭代预测t+3，以此迭代到t+23时刻(默认时间分辨率为6小时，20时刻共5天)；

medium版本使用的输入数据通过short预测得到，即通过short预测到t+23、t+24作为输入，预测t+25，以此迭代到t+46(共5天)；

long版本使用的输入数据通过medium得到，随后进行迭代预测。

short、medium、long微调时与graphcast同样，不断改变输入步长进行微调，以short为例，通过将迭代预测步长从2缓慢增长到12(即每次以第几步为最终目标)，再跳跃至20，实现微调，medium和long同理。

因此fuxi的整个训练流程与其他模型有所差异，具体顺序如下：

**训练base模型--->微调short模型--->推理short模型--->微调medium模型--->推理medium模型--->微调long模型--->推理long模型**；

base模型推理只需安排在训练base模型之后即可，无必要顺序；

所有训练(微调)脚本整合在一起，根据单机单卡(train_single_node_single_device.sh)、单机多卡(train_single_node_multi_device.sh)以及集群训练(train_via_slurm.sh)分为3个文件，执行方式如下：

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

通过修改#SBATCH -N 后面的数字指定节点数；

在3个训练脚本末尾均包含4个训练py文件，按照顺序依次执行并注释其余3个文件名即可；

默认提供的脚本中以训练base文件为例，注释其余3个训练脚本。

## 推理

fuxi推理需要按照一定顺序，具体参考上一节的顺序介绍，推理脚本末尾同样包含4个文件，按照先后顺序执行即可；

单机单卡推理，推理结果将存放在通目录result/下：

```
bash infer_single_node_single_device.sh
```

**误差计算**

base模型的误差可以通过下述脚本计算，方法为计算所有通道的RMSE及平均RMSE，并给出3个样本3个通道的结果图可视化：

```
python result.py
```

short、medium、long微调模型的误差通过result_three_mode.py计算，同样给出所有通道的RMSE及平均RMSE，并给出3个样本3个通道的结果图可视化：

```
python result_three_mode.py
```

## 模型快速部署测试方法

本节提供随机数据生成，便于用户快速部署模型，进行训练-推理测试。

```
import numpy as np
import h5py
import os
import sys
from onescience.utils.fcn.YParams import YParams

SHAPE, CHUNKS, DTYPE = (120, 70, 721, 1440), (1, 70, 721, 1440), "float32"
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
    arr = np.random.randn(1, 70, 1, 1).astype(np.float32)
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

生成数据后，由于训练+微调模型较多，下面按照执行顺序提供快速部署测试方法，主要依赖break断点退出来加速整个过程。

在train_fuxi_base.py的100行以及130行附近加入break，在160行附近加入exit()退出函数，能够检测训练正常进行以及快速跳过后续训练，便于保存模型；

train_fuxi_short.py、train_fuxi_medium.py、train_fuxi_long.py同理在相应位置加入2个break以及exit()，具体行号可能有所差异，但上下文基本一致，下述为相关代码段及前后文。

```
  ...
  ...
fuxi_model.train()
train_loss = 0
batch_start_time = time.time()
for j, data in enumerate(train_dataloader):
    if j == 10: # 新增代码便于快速测试
      	break
    invar = data[0].to(local_rank, dtype=torch.float32) # B, T, C, H, W
    invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
  ...
  ...
with torch.no_grad():
  for j, data in enumerate(val_dataloader):
      if j == 10: # 新增代码便于快速测试
          break
      invar = data[0].to(local_rank, dtype=torch.float32) # B, T, C, H, W
      invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
  ...
  ...
if valid_loss < best_valid_loss:
  best_valid_loss = valid_loss
  best_loss_epoch = epoch
  world_rank == 0 and save_checkpoint(
      fuxi_model,
      optimizer,
      scheduler,
      best_valid_loss,
      best_loss_epoch,
      cfg.checkpoint_dir,
  )
  is_save_ckp = True
  exit()# 新增代码便于快速测试
```

在inference_fuxi_short.py、inference_fuxi_medium.py、inference_fuxi_long.py的25行附近加入break，快速跳过重复迭代预测的过程，加速推理

```
  ...
  ...
invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
outvar = data[1].to('cuda:0', dtype=torch.float32)
for t in range(step):
    if t > 1:# 新增代码便于快速测试
        break
    outvar_pred = model(invar)
    invar[:, :, 0] = invar[:, :, -1]
    invar[:, :, -1] = outvar_pred
  ...
  ...
```

## 许可证

Fu xi项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。

