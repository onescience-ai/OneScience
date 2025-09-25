# <div align="center"><strong>Onescience for Evo2</strong></div>
## <div align="center">使用说明</div>


### 模型简介

Evo2 是一款面向基因组的基础模型，基于 **StripedHyena 2** 架构，支持**最长百万碱基上下文**，在大规模基因组数据集 **OpenGenome2** 上训练，覆盖细菌、古菌和真核等多类物种。模型提供 **7B** 和 **40B** 等版本，具备强大的长序列建模能力，可应用于变异效应预测、基因组设计和跨尺度序列分析。Evo2 已集成至本项目，支持高性能推理和微调，适合科研与实际生物学应用场景。

### 模型结构
![](../../../doc/evo2.jpg)

### 环境安装
```shell
conda create -n your-name python=3.11 -y
pip install packages of constraints.txt
pip install -c constraints.txt .[bio]
```

### 数据集准备
OpenGenome2 官方提供了两种格式的数据，该数据集大小约 2.5T，OpenGenome2[数据下载地址](https://modelscope.cn/datasets/arcinstitute/opengenome2)：
#### 1. 原始 FASTA 文件
  - 包含原始基因组序列，需要用户自行进行转录、反转录、序列互补、序列反转等预处理操作。
  - 适合需要 灵活处理 DNA 序列的研究场景。

**依赖工具**：  
- **`bionemo-noodles`**：基于 `noodles` 的 Python 封装，扩展了 **FAIDX (FASTA indexer)**，支持内存映射 (memmap)，可高效进行 FASTA 文件的随机访问。  
- 常用函数：  
  - `back_transcribe_sequence`: RNA → DNA 反转录  
  - `transcribe_sequence`: DNA → RNA 转录  
  - `complement_sequence`: DNA 序列互补链  
  - `reverse_sequence`: DNA 序列反转  

```shell
# shell 脚本
bash tools/data_process/preprocess_data_fasta.sh
# Python 脚本
python tools/data_process/preprocess_data_fasta.py -c <CONFIG_PATH>
```
#### 2. 预处理好的 JSON 文件
  - 官方已经对原始数据做了初步处理。
  - 仅需进行轻量级处理，例如数据读取、tokenizer 转换、样本长度填充（padding）等操作。
  - 适合快速实验。
```bash
python preprocess_data_json.py \
    --input "$INPUT_FILE" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type CharLevelTokenizer \
    --dataset-impl mmap \
    --append-eod \
    --enforce-sample-length 8192 \
    --workers 8 \
    --log-interval 100
```

### 模型转换

- 将单个 PyTorch 或 ZeRO-1 的 checkpoint（.pt 文件）转换为 NeMo2 格式
- 模型转化的脚本位置
  `onescience/examples/biosciences/evo2/tools/checkpoint_convert/convert_to_nemo.py`
- 实用示例
  `python tools/checkpoint_convert/convert_to_nemo.py --model-path <CKPT_FILE> --output-dir <OUTPUT_DIR>  --model-size <MODEL_SIZE>`

#### 7B 脚本示例
```bash
srun python tools/checkpoint_convert/convert_to_nemo.py \
  --model-path checkpoint/evo2_savanna_7b/savanna_evo2_7b.pt \
  --output-dir /work/share/ac8hkycjba/osmodels/evo2/nemo_model/nemo_evo2_7b \
  --model-size 7b_arc_longcontext 
```

#### 注意事项

1. **模型权重来源**  
   - 官网提供了两种模型权重：**训练** 和 **推理**。  
   - 请务必下载并使用 **训练用权重**（前缀为 `savanna_` 的模型权重）。  

   ![](../../../doc/evo2_model.png)

2. **`--model-size` 参数说明**  
   - 对于 7B 和 40B 的模型，需注意 `--model-size` 参数取值：  

   | 参数值               | 对应模型              |
   |----------------------|----------------------|
   | `7b`                 | `savanna_evo2_7b_base` |
   | `7b_arc_longcontext` | `savanna_evo2_7b`      |
   | `40b`                | `savanna_evo2_40b_base`|
   | `40b_arc_longcontext`| `savanna_evo2_40b`     |


 ### 训练
`onescience/examples/biosciences/evo2/checkpoint` 和 `onescience/examples/biosciences/evo2/data`分别用于存放模型与数据，可以通过软链接的方式将目标路径指向这里。

**单节点多卡训练**

1. 需要加载dtk相关环境(以612为例)：
    ```bash
    source ~/dtk/dtk-25.04.1/env.sh
    source ~/dtk/dtk-25.04.1/cuda/env.sh
    module load compiler/gcc/12.2.0
    ```
2. 运行脚本进行训练或微调
    ```bash
    # 从零训练只需注释掉 ckpt-dir 参数即可
    sh train_single_node_evo2_7b.sh
    ```
3. 重要参数说明
- 必要参数：训练脚本 `train_one_node.py`
- 必要参数：数据配置文件`the path of your data config`,具体格式可以参考config文件夹下示例
- dataset-dir：数据存放地址，和data config保持一致
- model-size：模型类型，可选有`1b,1b_nv,40b,40b_arc_longcontext,40b_nv,7b,7b_arc_longcontext,7b_nv,test,test_nv`
- devices：用到显卡数量
- ckpt-dir：预加载模型地址

```shell
python  $PROJECT_ROOT/examples/biosciences/evo2/train_one_node.py\
    -d $PROJECT_ROOT/examples/biosciences/evo2/config/genome_data_config.yaml\
    --dataset-dir $PROJECT_ROOT/examples/biosciences/evo2/data/genome_data\
    --model-size 7b_arc_longcontext\
    --devices 4 \
    --num-nodes 1 \
    --seq-length 8192 \
    --micro-batch-size 2 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 1000 \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 50 \
    --ckpt-async-save\
    # --ckpt-dir .model \
```

**多节点多卡训练**

多节点多卡主要涉及sbatch配置文件`train_multi_node_slurm_evo2.sh`和执行文件`train_evo2.sh`：
```shell
train_multi_node_slurm_evo2.sh

#!/bin/bash
#SBATCH -J evo2_for_onescience # 集群项目名字
#SBATCH -p k100ai # 申请显卡型号
#SBATCH --nodes=4 # 使用节点个数
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4 # 单节点使用显卡数
#SBATCH -o evo2/logs%j.out     # log地址，如需要二级目标，需要先手动建立文件夹 

# 612集群激活相关环境，508有所不同
source ~/dtk/dtk-25.04.1/env.sh
source ~/dtk/dtk-25.04.1/cuda/env.sh
module load compilers/gcc/12.2.0
source ~/conda.env
conda activate test-evo2env
unset ROCBLAS_TENSILE_LIBPATH 

DEVICES=${SLURM_GPUS_PER_NODE:-4}
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" 

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3 # 单节点卡数
export CUDA_DEVICE_MAX_CONNECTIONS=1

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# 第一个节点的地址
master_addr=${nodes_array[0]}

# 主节点的端口（可以根据需要调整）
master_port=29500

# 在每个节点上启动 torchrun
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port

srun train_evo2.sh

```
```shell
train_evo2.sh 相关参数含义参考单节点

#!/bin/bash

MODEL_SIZE=1b
CP_SIZE=1
TP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=2
GRAD_ACC_BATCHES=1
SEQ_LEN=512
MAX_STEPS=100
VAL_CHECK=50
CLIP_GRAD=250 # 梯度剪裁
EXTRA_ARGS="--enable-preemption --use-megatron-comm-overlap-llama3-8k --ckpt-async-save --overlap-grad-reduce --clip-grad $CLIP_GRAD --eod-pad-in-loss-mask"
EXTRA_ARG_DESC="BF16_perf_cg250_continue"
LR=0.0003
MIN_LR=0.00003
WU_STEPS=2500
# 0xDEADBEEF
SEED=1234
WD=0.1
ADO=0.01
HDO=0.01

# DEVICES=${SLURM_GPUS_PER_NODE:-4}
# echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
# echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" 

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[5])")

echo "ONESCIENCE_PATH:" $PROJECT_ROOT

cd $PROJECT_ROOT/examples/biosciences/evo2/checkpoint/evo2-7b

DIRS=(
    "./lightning_logs"
    "./results"
)

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Del Files: $DIR"
        rm -rf "$DIR"
    else
        echo "Files Not Exist: $DIR"
    fi
done

python $PROJECT_ROOT/examples/biosciences/evo2/train_slurm.py\
    -d $PROJECT_ROOT/examples/biosciences/evo2/config/training_data_config.yaml\
    --dataset-dir $PROJECT_ROOT/examples/biosciences/evo2/data/data_evo2_612\
    --model-size 7b_arc_longcontext \
    --devices 4 \
    --num-nodes 4 \
    --seq-length 1024 \
    --micro-batch-size 4 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 1000 \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 50 \
    --ckpt-async-save\
    # --num-nodes=${SLURM_JOB_NUM_NODES} \
    # --devices=${DEVICES} \
    # --grad-acc-batches $GRAD_ACC_BATCHES \
    # --max-steps=$MAX_STEPS \
    # --seed $SEED \
    # ${EXTRA_ARGS} \
    # --lr $LR \
    # --wd $WD \
    # --min-lr $MIN_LR \
    # --warmup-steps $WU_STEPS \
    # --attention-dropout $ADO \
    # --hidden-dropout $HDO \
    # --limit-val-batches=20 \
    # --val-check-interval=${VAL_CHECK} \
    # --seq-length=${SEQ_LEN} \
    # --tensor-parallel-size=${TP_SIZE} \
    # --context-parallel-size=${CP_SIZE} \
    # --pipeline-model-parallel-size=${PP_SIZE} \
    # --micro-batch-size=${MICRO_BATCH_SIZE} \
    # --model-size=${MODEL_SIZE} \
    # --workers 10

```

 ### 推理
在获得预训练或微调后的 **Evo2 checkpoint** 后，可以使用如下命令让模型根据提示生成 DNA 序列：
```bash
python infer.py --help 
```
**命令行参数说明**
```text
usage: infer_evo2 [-h] [--prompt PROMPT] --ckpt-dir CKPT_DIR
                  [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P]
                  [--max-new-tokens MAX_NEW_TOKENS]
                  [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                  [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE]
                  [--context-parallel-size CONTEXT_PARALLEL_SIZE]
                  [--output-file OUTPUT_FILE]
options:
  -h, --help            显示帮助信息并退出。
  --prompt PROMPT       用于生成序列的提示词。默认是大肠杆菌 (E. coli) 的系统发育分类标签。
  --ckpt-dir CKPT_DIR   指向包含预训练 Evo2 模型的 NeMo2 checkpoint 目录。（必填）
  --temperature TEMPERATURE
  --top-k TOP_K         
  --top-p TOP_P         
  --max-new-tokens MAX_NEW_TOKENS    生成的最大新 token 数。                   
  --tensor-parallel-size TENSOR_PARALLEL_SIZE    张量并行大小，默认值为 1。
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE    流水线并行大小，默认值为 1。         
  --context-parallel-size CONTEXT_PARALLEL_SIZE    上下文并行大小，默认值为 1。
  --output-file OUTPUT_FILE    生成序列的输出文件。如果未指定，输出将直接打印在终端。                   
```
**使用示例**
```bash
# 最简单的调用方式
srun python infer.py  --ckpt-dir checkpoint/evo2_nemo_7b --prompt "ATGCGT"
# 将输出结果保存为 .txt 文件
srun python infer.py  --ckpt-dir checkpoint/evo2_nemo_7b --prompt "ATGCGT" --output-file result.txt
```
**注意**   
--ckpt-dir 加载的 checkpoint 需要是 evo2 的 NeMo2 类型的checkpoint。

### 在超算互联网使用

### 许可证

evo2项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。

