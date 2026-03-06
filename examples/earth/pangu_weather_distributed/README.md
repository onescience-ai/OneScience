# Pangu-Weather

**模型简介**

华为云盘古气象大模型是首个精度超过传统数值预报方法的AI模型，速度相比传统数值预报提速10000倍以上。目前，盘古气象大模型能够提供全球气象秒级预报，其气象预测结果包括位势、湿度、风速、温度、海平面气压等，可以直接应用于多个气象研究细分场景，欧洲中期预报中心和中央气象台等都在实测中发现盘古预测的优越性。

**模型结构**

使用引入地球坐标的3D Transformer、Swin transformer以加速计算过程并且减少了网络深度和宽度以适配硬件。

**数据集准备**

曙光新一代机器平台数据集统一存放在 =  /public/onestore/onedatasets/ERA5

天津体验区统一存放在 = /work/home/onescience2025/osdatasets/pangu_weather/dataset 

用户如自备数据，则需在conf/config.yaml中指定数据路径，下述路径具体包含内容为：

```
stats_dir: #均值、标准差
mask_dir: #陆地掩码
checkpoint_dir: #模型文件存储路径
train_data_dir: #训练集
val_data_dir: #验证集
test_data_dir: #推理集
```

在训练阶段数据集目录内需确保存在static、stats、train、val四个文件夹；

在推理阶段数据集目录内需确保存在static、stats、test三个文件夹；


## 模型训练

参数设置约束：

--pipeline-model-parallel-size 流水线并行大小，该案例为2
--tensor-model-parallel-size 张量并行大小
--data-parallel-size=总卡数/--pipeline-model-parallel-size/--tensor-model-parallel-size
--hidden-size=config中embedding的维度
--num-attention-heads是--tensor-model-parallel-size的整数倍
--hidden-size是--num-attention-heads的整数倍
--global-batch-size是(--micro-batch-size*--data-parallel-size)的整数倍

单机多卡训练：

```
torchrun --nproc-per-node=8 train_distributed.py --micro-batch-size 1 --global-batch-size 1 --encoder-seq-length=1 --num-layers=4 --hidden-size=256 --num-attention-heads=8 --max-position-embeddings=1 --tokenizer-type=NullTokenizer --vocab-size=128 --train-iters 58400000 --lr 0.0001 --pipeline-model-parallel-size 2 --tensor-model-parallel-size 4 --eval-iters 5840 --fp16
```

多机多卡训练：

```bash
sbatch train_slurm_distributed.sh
```

运行多机多卡训练前，需确保目录内有logs文件(默认没有该文件夹)

默认每个节点有8卡；

通过修改#SBATCH -N 后面的数字指定节点数，下面以个2节点为例，每个节点8卡，共16卡；

若没有该文件，则参考下述命令手动创建即可；

conda激活的环境需自行指定。

```
#!/bin/bash
#SBATCH -p largedev
#SBATCH -N 4
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J pangu_weather
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.out

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate pangu_weather
module load sghpc-mpi-gcc/25.8

which python
which hipcc

export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0 
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=16
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

export MASTER_ADDR=$(hostname)

master_port=29500

echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$MASTER_ADDR
echo master_port=$master_port
echo "Nodes: ${nodes_array[*]}"

srun -u --mpi=pmix \
    bash -c "
    source export_DDP_vars.sh
    python train_distributed.py --micro-batch-size 1 --global-batch-size 1 --encoder-seq-length=1 --num-layers=4 --hidden-size=256 \
            --num-attention-heads=8 --max-position-embeddings=1 --tokenizer-type=NullTokenizer --vocab-size=128 --lr-decay-style linear \
            --train-iters 58400000 --lr 0.0001 --pipeline-model-parallel-size 2 --tensor-model-parallel-size 8 \
            --eval-interval 5840 --eval-iters 584 \
            --clip-grad 3.0 \
            --attention-dropout 0.1 --hidden-dropout 0.2 \
            --init-method-std 0.01 \
            --weight-decay 0.001 --adam-beta2 0.95 \
            --log-interval 1 \
            --fp16 \
            --initial-loss-scale 8192 \
    "
```

## 许可证

Pangu_weather项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。