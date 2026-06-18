# AlphaGenome

## 概述

AlphaGenome 是 Google DeepMind 开发的统一 DNA 序列基因组模型，能够以单碱基对分辨率分析长达 **100 万碱基对 (1 Mbp)** 的 DNA 序列，同时预测多种基因组功能信号。

本目录提供了 AlphaGenome 在 OneScience 框架下的示例脚本，涵盖推理、变异效应预测、性能评估和微调四个核心场景。AlphaGenome 的模型实现已集成至 `onescience.flax_models.alphagenome` 命名空间。

### 主要能力

| 预测类型 | 数据类型 | 分辨率 |
|---------|---------|--------|
| 染色质可及性 | ATAC-seq, DNase-seq | 1 bp |
| 转录起始位点 | CAGE, PRO-cap | 1 bp |
| 基因表达 | RNA-seq | 1 bp |
| 转录因子结合 | ChIP-seq (TF) | 128 bp |
| 组蛋白修饰 | ChIP-seq (Histone) | 128 bp |
| 三维基因组结构 | Hi-C 接触图谱 | 2048 bp |
| 剪接位点 | 剪接位点分类/使用率/连接对 | 1 bp |

支持物种：人类 (*Homo sapiens* / hg38) 和小鼠 (*Mus musculus* / mm10)


## 示例脚本

### 1. 推理 (`run_inference.py`)

对指定基因组区间运行 AlphaGenome 推理，输出所有支持的基因组功能预测。

```bash
python run_inference.py \
    --fasta_path ${DATA_ROOT_DIR}/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --chromosome chr19 \
    --start 10587331 \
    --end 11635907 \
    --output_dir ./outputs
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--fasta_path` | 参考基因组 FASTA 文件 (需 .fai 索引) | 无 (Kaggle 模式自动配置) |
| `--model_dir` | 模型权重目录 | 自动从 Kaggle Hub 下载 |
| `--chromosome` | 染色体名称 | `chr1` |
| `--start` | 区间起始位置 (0-based) | `1000000` |
| `--end` | 区间终止位置 | `2048576` |
| `--output_dir` | 输出目录 | `./outputs` |
| `--organism` | 生物体 (`HOMO_SAPIENS`/`MUS_MUSCULUS`) | `HOMO_SAPIENS` |
| `--model_version` | 模型版本 (`FOLD_0`~`FOLD_4` 或 `all_folds`) | `FOLD_0` |

**输出：** 每种预测类型保存为独立的 `.npy` 文件。

**运行结果示例：**

```text
I0519 09:40:43.747026 140521812137792 run_inference.py:91] 加载模型权重...
I0519 09:40:45.827422 140521812137792 xla_bridge.py:895] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
sh: /root/miniconda3/envs/gaozhan/lib/libtinfo.so.6: no version information available (required by sh)
I0519 09:40:54.180782 140521812137792 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.5
I0519 09:40:54.181102 140521812137792 base_pytree_checkpoint_handler.py:332] Created BasePyTreeCheckpointHandler: pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=None
I0519 09:40:54.181279 140521812137792 async_checkpointer.py:83] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>.<lambda> at 0x7fcb4fdc4040> timeout: 600 secs and primary_host=0 for async checkpoint writes
I0519 09:40:54.181736 140521812137792 checkpointer.py:277] Restoring checkpoint from /root/private_data/gaozhan/OneScience_Test/all-folds.
I0519 09:40:55.409961 140521812137792 base_pytree_checkpoint_handler.py:113] [process=0] /jax/checkpoint/read/bytes_per_sec: 1.4 GiB/s (total bytes: 1.7 GiB) (time elapsed: a second) (per-host)
I0519 09:40:55.411144 140521812137792 checkpointer.py:288] Finished restoring checkpoint in 1.23 seconds from /root/private_data/gaozhan/OneScience_Test/all-folds.
I0519 09:40:56.173897 140521812137792 run_inference.py:110] 运行推理: chr19:10587331-11635907:.
I0519 09:41:27.682349 140521812137792 run_inference.py:146] 保存预测结果至 outputs
I0519 09:41:28.766666 140521812137792 run_inference.py:153]   保存 atac: shape=(1048576, 167)
I0519 09:41:31.129668 140521812137792 run_inference.py:153]   保存 dnase: shape=(1048576, 305)
I0519 09:41:35.282252 140521812137792 run_inference.py:153]   保存 cage: shape=(1048576, 546)
I0519 09:41:41.164901 140521812137792 run_inference.py:153]   保存 rna_seq: shape=(1048576, 667)
I0519 09:41:41.262393 140521812137792 run_inference.py:153]   保存 chip_tf: shape=(8192, 1617)
I0519 09:41:41.329550 140521812137792 run_inference.py:153]   保存 chip_histone: shape=(8192, 1116)
I0519 09:41:41.329753 140521812137792 run_inference.py:156] 推理完成。
```

---

### 2. 变异效应预测 (`run_variant_scoring.py`)

计算单核苷酸变异 (SNV) 对基因组功能的影响，返回多种评分表 (AnnData 格式)。

```bash
# 使用内置演示变异
python run_variant_scoring.py \
    --fasta_path ${DATA_ROOT_DIR}/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --output_dir ./outputs_variant

# 使用自定义 VCF 文件
python run_variant_scoring.py \
    --fasta_path ${DATA_ROOT_DIR}/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --vcf_path /path/to/variants.vcf \
    --output_dir ./outputs/variants
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--vcf_path` | VCF 变异文件 | 无 (使用内置演示变异) |
| `--fasta_path` | 参考基因组 FASTA 文件 (需 .fai 索引) | 无 |
| `--model_dir` | 本地模型权重目录 | 未指定时从 Kaggle Hub 加载 |
| `--output_dir` | 输出目录 | `./outputs` |
| `--organism` | 生物体 | `HOMO_SAPIENS` |
| `--model_version` | 模型版本 | `all_folds` |

**输出：** 每个变异的评分表 (CSV) 和汇总文件 `variant_scoring_summary.csv`。

**运行结果示例：**

```text
I0519 09:52:20.181227 140547641341760 run_variant_scoring.py:203] Using built-in demo variants.
I0519 09:52:20.181590 140547641341760 run_variant_scoring.py:206] Scoring 3 variants...
I0519 09:52:20.181660 140547641341760 run_variant_scoring.py:210] Processing variant: chr22:36201698:A>C (eQTL with SuSiE PIP > 0.9 in GTEx Colon)
I0519 09:53:02.841866 140547641341760 run_variant_scoring.py:239] Saved score table 0: variant_chr22_36201698_CenterMaskScorer(requested_output=ATAC,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 167))
I0519 09:53:02.846082 140547641341760 run_variant_scoring.py:239] Saved score table 1: variant_chr22_36201698_ContactMapScorer().csv (shape=(1, 28))
I0519 09:53:02.849872 140547641341760 run_variant_scoring.py:239] Saved score table 2: variant_chr22_36201698_CenterMaskScorer(requested_output=DNASE,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 305))
I0519 09:53:02.857878 140547641341760 run_variant_scoring.py:239] Saved score table 3: variant_chr22_36201698_CenterMaskScorer(requested_output=CHIP_TF,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 1617))
I0519 09:53:02.864007 140547641341760 run_variant_scoring.py:239] Saved score table 4: variant_chr22_36201698_CenterMaskScorer(requested_output=CHIP_HISTONE,_width=2001,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 1116))
I0519 09:53:02.868129 140547641341760 run_variant_scoring.py:239] Saved score table 5: variant_chr22_36201698_CenterMaskScorer(requested_output=CAGE,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 546))
I0519 09:53:02.873609 140547641341760 run_variant_scoring.py:239] Saved score table 6: variant_chr22_36201698_CenterMaskScorer(requested_output=PROCAP,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 12))
I0519 09:53:02.877482 140547641341760 run_variant_scoring.py:239] Saved score table 7: variant_chr22_36201698_CenterMaskScorer(requested_output=ATAC,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 167))
I0519 09:53:02.881478 140547641341760 run_variant_scoring.py:239] Saved score table 8: variant_chr22_36201698_CenterMaskScorer(requested_output=DNASE,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 305))
I0519 09:53:02.889644 140547641341760 run_variant_scoring.py:239] Saved score table 9: variant_chr22_36201698_CenterMaskScorer(requested_output=CHIP_TF,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 1617))
I0519 09:53:02.896199 140547641341760 run_variant_scoring.py:239] Saved score table 10: variant_chr22_36201698_CenterMaskScorer(requested_output=CHIP_HISTONE,_width=2001,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 1116))
I0519 09:53:02.900317 140547641341760 run_variant_scoring.py:239] Saved score table 11: variant_chr22_36201698_CenterMaskScorer(requested_output=CAGE,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 546))
I0519 09:53:02.902701 140547641341760 run_variant_scoring.py:239] Saved score table 12: variant_chr22_36201698_CenterMaskScorer(requested_output=PROCAP,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 12))
I0519 09:53:02.902797 140547641341760 run_variant_scoring.py:210] Processing variant: chr3:120280774:G>T (caQTL in GM12878 (DNase))
I0519 09:53:15.194623 140547641341760 run_variant_scoring.py:239] Saved score table 0: variant_chr3_120280774_CenterMaskScorer(requested_output=ATAC,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 167))
I0519 09:53:15.197259 140547641341760 run_variant_scoring.py:239] Saved score table 1: variant_chr3_120280774_ContactMapScorer().csv (shape=(1, 28))
I0519 09:53:15.200717 140547641341760 run_variant_scoring.py:239] Saved score table 2: variant_chr3_120280774_CenterMaskScorer(requested_output=DNASE,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 305))
I0519 09:53:15.208721 140547641341760 run_variant_scoring.py:239] Saved score table 3: variant_chr3_120280774_CenterMaskScorer(requested_output=CHIP_TF,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 1617))
I0519 09:53:15.215613 140547641341760 run_variant_scoring.py:239] Saved score table 4: variant_chr3_120280774_CenterMaskScorer(requested_output=CHIP_HISTONE,_width=2001,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 1116))
I0519 09:53:15.221033 140547641341760 run_variant_scoring.py:239] Saved score table 5: variant_chr3_120280774_CenterMaskScorer(requested_output=CAGE,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 546))
I0519 09:53:15.224044 140547641341760 run_variant_scoring.py:239] Saved score table 6: variant_chr3_120280774_CenterMaskScorer(requested_output=PROCAP,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 12))
I0519 09:53:15.227978 140547641341760 run_variant_scoring.py:239] Saved score table 7: variant_chr3_120280774_CenterMaskScorer(requested_output=ATAC,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 167))
I0519 09:53:15.231396 140547641341760 run_variant_scoring.py:239] Saved score table 8: variant_chr3_120280774_CenterMaskScorer(requested_output=DNASE,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 305))
I0519 09:53:15.239303 140547641341760 run_variant_scoring.py:239] Saved score table 9: variant_chr3_120280774_CenterMaskScorer(requested_output=CHIP_TF,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 1617))
I0519 09:53:15.245375 140547641341760 run_variant_scoring.py:239] Saved score table 10: variant_chr3_120280774_CenterMaskScorer(requested_output=CHIP_HISTONE,_width=2001,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 1116))
I0519 09:53:15.249743 140547641341760 run_variant_scoring.py:239] Saved score table 11: variant_chr3_120280774_CenterMaskScorer(requested_output=CAGE,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 546))
I0519 09:53:15.252806 140547641341760 run_variant_scoring.py:239] Saved score table 12: variant_chr3_120280774_CenterMaskScorer(requested_output=PROCAP,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 12))
I0519 09:53:15.252902 140547641341760 run_variant_scoring.py:210] Processing variant: chr21:46126238:G>C (Splice junction variant in COL6A2)
I0519 09:53:27.369731 140547641341760 run_variant_scoring.py:239] Saved score table 0: variant_chr21_46126238_CenterMaskScorer(requested_output=ATAC,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 167))
I0519 09:53:27.374533 140547641341760 run_variant_scoring.py:239] Saved score table 1: variant_chr21_46126238_ContactMapScorer().csv (shape=(1, 28))
I0519 09:53:27.379677 140547641341760 run_variant_scoring.py:239] Saved score table 2: variant_chr21_46126238_CenterMaskScorer(requested_output=DNASE,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 305))
I0519 09:53:27.391176 140547641341760 run_variant_scoring.py:239] Saved score table 3: variant_chr21_46126238_CenterMaskScorer(requested_output=CHIP_TF,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 1617))
I0519 09:53:27.399726 140547641341760 run_variant_scoring.py:239] Saved score table 4: variant_chr21_46126238_CenterMaskScorer(requested_output=CHIP_HISTONE,_width=2001,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 1116))
I0519 09:53:27.405907 140547641341760 run_variant_scoring.py:239] Saved score table 5: variant_chr21_46126238_CenterMaskScorer(requested_output=CAGE,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 546))
I0519 09:53:27.410531 140547641341760 run_variant_scoring.py:239] Saved score table 6: variant_chr21_46126238_CenterMaskScorer(requested_output=PROCAP,_width=501,_aggregation_type=DIFF_LOG2_SUM).csv (shape=(1, 12))
I0519 09:53:27.414916 140547641341760 run_variant_scoring.py:239] Saved score table 7: variant_chr21_46126238_CenterMaskScorer(requested_output=ATAC,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 167))
I0519 09:53:27.420156 140547641341760 run_variant_scoring.py:239] Saved score table 8: variant_chr21_46126238_CenterMaskScorer(requested_output=DNASE,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 305))
I0519 09:53:27.429220 140547641341760 run_variant_scoring.py:239] Saved score table 9: variant_chr21_46126238_CenterMaskScorer(requested_output=CHIP_TF,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 1617))
I0519 09:53:27.437276 140547641341760 run_variant_scoring.py:239] Saved score table 10: variant_chr21_46126238_CenterMaskScorer(requested_output=CHIP_HISTONE,_width=2001,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 1116))
I0519 09:53:27.441566 140547641341760 run_variant_scoring.py:239] Saved score table 11: variant_chr21_46126238_CenterMaskScorer(requested_output=CAGE,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 546))
I0519 09:53:27.445191 140547641341760 run_variant_scoring.py:239] Saved score table 12: variant_chr21_46126238_CenterMaskScorer(requested_output=PROCAP,_width=501,_aggregation_type=ACTIVE_SUM).csv (shape=(1, 12))
I0519 09:53:27.448495 140547641341760 run_variant_scoring.py:245] Saved summary to outputs_variant/variant_scoring_summary.csv
```

---

### 3. 轨迹预测评估 (`run_track_prediction_eval.py`)

在官方验证集上评估模型的基因组轨迹预测性能。

```bash
# 使用本地权重评估 (HOMO_SAPIENS, ALL_FOLDS, 所有数据束)
python run_track_prediction_eval.py \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --model_version ALL_FOLDS \
    --output_path ./outputs_track/eval_results.csv

# 评估特定数据束
python run_track_prediction_eval.py \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --model_version FOLD_1 \
    --organism MUS_MUSCULUS \
    --bundles ATAC,CAGE,DNASE \
    --output_path ./results/eval_fold1_mouse.csv
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_version` | 模型版本 | `FOLD_0` |
| `--organism` | 生物体 | `HOMO_SAPIENS` |
| `--model_dir` | 本地模型权重目录 | 未指定时拒绝下载 |
| `--allow_download` | 未提供 `--model_dir` 时是否允许从 Kaggle Hub 下载权重 | `False` |
| `--data_dir` | 本地 AlphaGenome TFRecord 数据目录 | 未指定时使用数据加载器默认路径 |
| `--bundles` | 评估数据束 (逗号分隔) | 所有支持的数据束 |
| `--output_path` | 结果 CSV 路径 | `./track_prediction_results.csv` |

**输出：** CSV 文件，包含 `bundle`、`metric`、`value` 列。

**注意：** 评估数据默认由数据加载器读取；若本地没有数据，可通过 `--data_dir` 指定本地 TFRecord 根目录。完整评估可能需要数小时。

**运行结果示例：**

```text
I0518 23:17:47.833750 140248491935552 run_track_prediction_eval.py:253] step 1: {'ATAC': {'mae': Array(0.0421307, dtype=float32),
          'mse': Array(0.03742046, dtype=float32),
          'pearsonr': 0.3926131,
          'pearsonr_log1p': 0.37209284},
 'CAGE': {'mae': Array(0.00290494, dtype=float32),
          'mse': Array(0.08423273, dtype=float32),
          'pearsonr': 0.011533643,
          'pearsonr_log1p': 0.012083948},
 'CHIP_HISTONE': {'mae': Array(33.256836, dtype=float32),
                  'mse': Array(2578.609, dtype=float32),
                  'pearsonr': 0.4388966,
                  'pearsonr_log1p': 0.42592978},
 'CHIP_TF': {'mae': Array(38.092087, dtype=float32),
             'mse': Array(3096.2258, dtype=float32),
             'pearsonr': 0.45927477,
             'pearsonr_log1p': 0.4389368},
 'DNASE': {'mae': Array(0.02584387, dtype=float32),
           'mse': Array(0.01945999, dtype=float32),
           'pearsonr': 0.3492914,
           'pearsonr_log1p': 0.3153439},
 'PROCAP': {'mae': Array(0.00074321, dtype=float32),
            'mse': Array(0.00145492, dtype=float32),
            'pearsonr': 0.227709,
            'pearsonr_log1p': 0.19063582},
 'RNA_SEQ': {'mae': Array(0.00016873, dtype=float32),
             'mse': Array(1.1348125e-05, dtype=float32),
             'pearsonr': 0.057020403,
             'pearsonr_log1p': 0.057204794}}
```

---

### 4. 微调 (`run_finetuning.py`)

在自定义数据上微调 AlphaGenome 预训练模型。

```bash
python run_finetuning.py \
    --fasta_path /path/to/GRCh38.p13.genome.fa \
    --regions_csv /path/to/training_regions.csv \
    --bigwig_paths /path/to/my_atac.bw,/path/to/my_atac2.bw \
    --output_dir ./finetuned_model \
    --num_steps 1000 \
    --batch_size 2 \
    --learning_rate 1e-5
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--fasta_path` | 参考基因组 FASTA 文件 | 必需 |
| `--regions_csv` | 训练区间 CSV (列: chromosome,start,end) | 必需 |
| `--bigwig_paths` | BigWig 信号文件列表 | 必需 |
| `--model_dir` | 预训练权重目录 | 自动下载 |
| `--output_dir` | 保存目录 | `./finetuned_model` |
| `--num_steps` | 训练步数 | `1000` |
| `--batch_size` | 批次大小 | `2` |
| `--learning_rate` | 学习率 | `1e-5` |
| `--log_every` | 日志频率 (步) | `50` |
| `--save_every` | 检查点频率 (步) | `200` |

**`regions.csv` 格式示例：**
```csv
chromosome,start,end
chr1,1000000,2048576
chr2,5000000,6048576
```

---

## 在 OneScience 中的使用

AlphaGenome 已集成至 `onescience.flax_models.alphagenome` 命名空间，可直接导入使用：

### 模型推理

```python
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome._sdk.models import dna_model
from onescience.flax_models.alphagenome.model.dna_model import create_from_kaggle

# 加载模型
model = create_from_kaggle('all_folds')

# 定义区间并预测
interval = genome.Interval(chromosome='chr19', start=1_058_7331, end=1_163_5907)
predictions = model.predict_interval(
    interval,
    organism=dna_model.Organism.HOMO_SAPIENS,
    requested_outputs={
        dna_model.OutputType.RNA_SEQ,
        dna_model.OutputType.ATAC,
        dna_model.OutputType.DNASE,
    },
    ontology_terms=None,
)
```

### 变异效应预测

```python
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome._sdk.models import dna_model
from onescience.flax_models.alphagenome.model.dna_model import create_from_kaggle

model = create_from_kaggle('all_folds')
variant = genome.Variant.from_str('chr22:36201698:A>C')
interval = variant.reference_interval.resize(2**20)

# predict_variant 返回 VariantOutput (含 .reference 和 .alternate)
outputs = model.predict_variant(
    interval, variant,
    requested_outputs=[dna_model.OutputType.RNA_SEQ],
    ontology_terms=['UBERON:0001159'],
)

# score_variant 返回 list[AnnData] (变异效应分数)
scores = model.score_variant(interval, variant)
```

### 数据加载

```python
from onescience.flax_models.alphagenome.io.bundles import BundleName
from onescience.flax_models.alphagenome.io.dataset import get_numpy_dataset_iterator
from onescience.flax_models.alphagenome._sdk.data import fold_intervals
from onescience.flax_models.alphagenome._sdk.models import dna_model

iterator = get_numpy_dataset_iterator(
    batch_size=4,
    organism=dna_model.Organism.HOMO_SAPIENS,
    model_version=dna_model.ModelVersion.FOLD_0,
    subset=fold_intervals.Subset.VALID,
    bundles=[BundleName.ATAC, BundleName.CAGE, BundleName.RNA_SEQ],
)

for batch, metadata in iterator:
    print(batch.dna_sequence.shape)  # (4, 1048576, 4)
    print(batch.atac.shape)          # (4, 1048576, n_atac_tracks)
```

### DNA 序列编码

```python
from onescience.flax_models.alphagenome.model.one_hot_encoder import DNAOneHotEncoder

encoder = DNAOneHotEncoder()
encoded = encoder.encode('ATCGNATCG')  # shape: (9, 4)
# A=[1,0,0,0], T=[0,0,0,1], C=[0,1,0,0], G=[0,0,1,0], N=[0,0,0,0]
```

### 微调训练步骤

```python
import optax
from onescience.flax_models.alphagenome.finetuning.finetune import get_train_step

optimizer = optax.adam(learning_rate=1e-5)
train_step = get_train_step(predict_fn=predict_fn, optimizer=optimizer)

for batch in dataset_iterator:
    params, state, opt_state, metrics = train_step(params, state, opt_state, batch)
    print(f"loss: {metrics['loss']:.4f}")
```

---

## 模块结构

AlphaGenome 在 OneScience 中的模块组织如下：

```
onescience/flax_models/alphagenome/
├── __init__.py                  # 包入口，版本信息
├── model/
│   ├── __init__.py
│   ├── model.py                 # AlphaGenome, SequenceEncoder, SequenceDecoder, TransformerTower
│   ├── layers.py                # gelu, pool, RMSBatchNorm, LayerNorm
│   ├── attention.py             # MHABlock, AttentionBiasBlock, MLPBlock, PairUpdateBlock, ...
│   ├── convolutions.py          # ConvBlock, DnaEmbedder, DownResBlock, UpResBlock
│   ├── embeddings.py            # Embeddings, OutputEmbedder, OutputPair
│   ├── heads.py                 # HeadName, HeadType, GenomeTracksHead, ContactMapsHead, ...
│   ├── losses.py                # poisson_loss, multinomial_loss, binary_crossentropy_from_logits, ...
│   ├── schemas.py               # DataBatch
│   ├── splicing.py              # generate_splice_site_positions
│   ├── one_hot_encoder.py       # DNAOneHotEncoder
│   ├── dna_model.py             # AlphaGenomeModel, create_from_kaggle, create, OrganismSettings, ...
│   ├── augmentation.py          # reverse_complement, reverse_complement_output_type
│   ├── variant_scoring/         # 变异效应评分
│   │   ├── variant_scoring.py   # VariantScorer (基类), create_anndata, get_resolution
│   │   ├── center_mask.py       # CenterMaskVariantScorer
│   │   ├── contact_map.py       # ContactMapScorer
│   │   ├── gene_mask.py         # GeneVariantScorer
│   │   ├── gene_mask_extractor.py # GeneMaskExtractor, GeneMaskType, GeneQueryType
│   │   ├── polyadenylation.py   # PolyadenylationVariantScorer
│   │   └── splice_junction.py   # SpliceJunctionVariantScorer
│   ├── interval_scoring/        # 区间评分
│   │   ├── interval_scoring.py  # IntervalScorer (基类)
│   │   └── gene_mask.py         # GeneIntervalScorer
│   └── metadata/                # 输出元数据
│       └── metadata.py          # AlphaGenomeOutputMetadata, create_track_masks, load
├── io/
│   ├── __init__.py
│   ├── bundles.py               # BundleName 枚举 (ATAC/CAGE/RNA_SEQ/...)
│   ├── dataset.py               # create_dataset, get_numpy_dataset_iterator
│   ├── fasta.py                 # FastaExtractor, reverse_complement
│   ├── genome.py                # extract_variant_sequences, insert_reference/alternate_variant
│   └── splicing.py              # PositionExtractor, SpliceSiteAnnotationExtractor
├── finetuning/
│   ├── __init__.py
│   ├── finetune.py              # get_dataset_iterator, get_forward_fn, get_train_step
│   └── dataset.py               # BigWigExtractor, MultiTrackExtractor, DataPipeline
└── evals/
    ├── __init__.py
    ├── regression_metrics.py    # RegressionState, initialize/update/finalize/reduce_regression_metrics
    └── track_prediction.py      # load_model, create_eval_step, evaluate, run
```

---

## 模型架构

AlphaGenome 采用 U-Net + Transformer 的混合架构：

```
DNA 序列 (1 Mbp × 4)
    │
    ▼
[DnaEmbedder]          卷积嵌入: 4 → 768 通道
    │
    ▼
[DownResBlock × 7]     卷积下采样: 1bp → 128bp 分辨率, 768→1536 通道
    │
    ├─────────────────── 跳跃连接 (U-Net skip)
    ▼
[TransformerTower × 9] 序列注意力 (MHABlock + MLPBlock + PairUpdateBlock)
    │
    ▼
[SequenceToPairBlock]  序列 → 成对表示投影
    │
    ├─── embeddings_pair (2048bp 分辨率)
    ▼
[UpResBlock × 7]       卷积上采样 (融合跳跃连接): 128bp → 1bp
    │
    ├─── embeddings_1bp  (1bp 分辨率, 1536 通道)
    └─── embeddings_128bp (128bp 分辨率, 3072 通道)
         │
         ▼
    [OutputEmbedder]   生物体特异性调整
         │
    ┌────┴──────────────────────────────────────┐
    │                                           │
[GenomeTracksHead]                    [ContactMapsHead]
  ATAC/DNase/CAGE/RNA-seq/              Hi-C 接触图谱
  PRO-cap/ChIP-TF/ChIP-Histone
    │
[SpliceSitesHead × 3]
  分类/使用率/连接对
```

---

## 引用

如果您在研究中使用了 AlphaGenome，请引用：

```bibtex
@article{alphagenome2025,
  title   = {AlphaGenome: advancing regulatory genomics with a unified DNA sequence model},
  author  = {Google DeepMind AlphaGenome Team},
  journal = {bioRxiv},
  year    = {2025},
  url     = {https://github.com/google-deepmind/alphagenome}
}
```

## 许可证

AlphaGenome 源代码基于 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 授权。

模型权重受独立的使用条款约束，详见 [AlphaGenome 模型权重使用条款](https://github.com/google-deepmind/alphagenome)。

---

## 相关链接

- [AlphaGenome 官方仓库](https://github.com/google-deepmind/alphagenome)
- [alphagenome PyPI 包](https://pypi.org/project/alphagenome/)
- [模型权重 (Kaggle Hub)](https://www.kaggle.com/models/google-deepmind/alphagenome)
- [数据集 (Google Cloud Storage)](https://console.cloud.google.com/storage/browser/alphagenome-datasets)
- [OneScience 项目主页](https://github.com/google-deepmind/onescience)
