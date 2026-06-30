# MedGemma 示例

本示例将 MedGemma 集成到 OneScience 生物信息（AI for Biology）组件中，提供面向医学场景的统一推理、微调与评估入口。

## MedGemma 简介

MedGemma 是 Google 开源的医学多模态大语言模型，基于 [Gemma 3](https://ai.google.dev/gemma/docs/core) 架构，针对医学文本与医学影像理解进行训练。MedGemma 提供两种变体：

- **MedGemma 4B**：多模态模型，支持医学文本与医学图像联合输入。
- **MedGemma 27B**：纯文本模型，专注于医学文本理解与问答。

MedGemma 4B 使用 [SigLIP](https://arxiv.org/abs/2303.15343) 图像编码器，已在多种去标识化医学数据上预训练，包括胸片（CXR）、皮肤科图像、眼科图像和组织病理学切片；其语言模型组件在放射学图像、病理学图像、眼科图像、皮肤科图像和医学文本上进行了训练。

MedGemma 已在多项临床相关基准上评估，涵盖开放基准数据集和专家人工评估任务。更多信息请参阅：

- [开发者文档](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [模型卡（Model Card）](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Hugging Face 模型](https://huggingface.co/models?other=medgemma)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)

当前示例默认基于 `google/medgemma-1.5-4b-it`（4B 多模态指令模型），支持文本与医学图像联合输入，可在 GPU 或海光 DCU 平台上运行。

---

## 目录

- [功能定位](#功能定位)
- [环境准备](#环境准备)
- [数据与模型权重](#数据与模型权重)
- [脚本速查表](#脚本速查表)
- [详细使用说明](#详细使用说明)
  - [1. 集成测试](#1-集成测试)
  - [2. 医学问答评估（`run_evaluate_on_medqa.sh`）](#2-医学问答评估run_evaluate_on_medqash)
  - [3. 胸片解剖结构定位（`run_cxr_anatomy.sh`）](#3-胸片解剖结构定位run_cxr_anatomysh)
  - [4. 胸片纵向对比分析（`run_cxr_longitudinal_comparison.sh`）](#4-胸片纵向对比分析run_cxr_longitudinal_comparisonsh)
  - [5. 病理图像 LoRA 微调（`run_fine_tune.sh`）](#5-病理图像-lora-微调run_fine_tunesh)
  - [6. 使用推理运行器](#6-使用推理运行器)
  - [7. Python API 调用](#7-python-api-调用)
- [目录结构](#目录结构)
- [注意事项](#注意事项)
- [许可证与引用](#许可证与引用)

---

## 功能定位

- **医学问答**：基于 MedQA 等医学知识基准评估模型问答能力。
- **医学影像分析**：支持胸片（CXR）解剖结构定位、多期影像对比分析等任务。
- **领域微调**：基于 NCT 结肠组织病理图像等数据，使用 LoRA 进行参数高效微调。
- **统一推理接口**：通过 `MedicalInferenceRunner` 提供交互式与批量文件推理能力。

---

## 环境准备

1. 参照项目根目录 [README.md](../../../README.md) 完成 OneScience（bio 领域）安装：

    ```bash
    bash install.sh bio
    ```

2. 激活环境：

    ```bash
    conda activate onescience311
    注：(1) 检查botocore的版本，若版本过低，请升级版本
    例如: pip install --upgrade boto3==1.43.36 botocore==1.43.36
    (2) 检查transformers版本，若版本过低，请升级版本
    例如：pip install --upgrade transformers==5.12.1
    ```

3. 确保 `ONESCIENCE_DATASETS_DIR` 环境变量已设置（通常由项目根目录 `env.sh` 自动配置）：

    ```bash
    source /path/to/onescience/env.sh
    ```

---

## 数据与模型权重

### 1. 模型权重

脚本默认从以下路径加载模型：

```
${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it
```

请提前下载模型并放置到该目录，或通过 `model_path` 环境变量覆盖。模型可通过以下渠道获取：

- [Hugging Face - google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- [ModelScope](https://modelscope.cn/)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)

### 2. 数据集

| 任务 | 数据 | 默认路径 |
|------|------|----------|
| MedQA 评估 | MedQA parquet 数据 | `${ONESCIENCE_DATASETS_DIR}/medgemma/medqa` |
| 胸片解剖定位 | 胸片图像 | `${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/...` |
| 胸片纵向对比 | 前后两次胸片 | `${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/...` |
| 病理图像微调 | NCT-CRC-HE-100K / CRC-VAL-HE-7K | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/...` |

数据集可通过以下方式获取：

- **MedQA**：https://github.com/jind11/MedQA
- **Chest X-ray**：推荐使用公开胸片数据集，如 COVID-19 Chest X-Ray Dataset 或 MIMIC-CXR
- **NCT-CRC-HE-100K / CRC-VAL-HE-7K**：https://zenodo.org/records/1214456

---

## 脚本速查表

以下 4 个 `bash` 脚本为本示例的官方入口，均可直接运行。

| 脚本 | 功能 | 推荐运行方式 | 输出目录 |
|------|------|--------------|----------|
| `scripts/run_evaluate_on_medqa.sh` | MedQA 医学问答评估 | `bash scripts/run_evaluate_on_medqa.sh` | `scripts/medqa_results/` |
| `scripts/run_cxr_anatomy.sh` | 胸片解剖结构定位（单图 + 批量） | `bash scripts/run_cxr_anatomy.sh` | `scripts/outputs/` |
| `scripts/run_cxr_longitudinal_comparison.sh` | 胸片前后对比分析 | `bash scripts/run_cxr_longitudinal_comparison.sh` | `scripts/compare_outputs/` |
| `scripts/run_fine_tune.sh` | NCT 病理图像 LoRA 微调 | `bash scripts/run_fine_tune.sh` | `scripts/medgemma-nct-lora/` |

以上脚本均位于 `examples/biosciences/medgemma/scripts/` 目录。

---

## 详细使用说明

所有脚本默认在 `examples/biosciences/medgemma/scripts/` 目录下执行，并自动定位项目根目录加载 `env.sh`。

### 1. 集成测试

验证 MedGemma 在 OneScience 中的模块、配置、数据适配器与图像处理组件是否可正常导入：

```bash
cd examples/biosciences/medgemma
python tests/test_integration.py
```

---

### 2. 医学问答评估（`run_evaluate_on_medqa.sh`）

在 MedQA 数据集上评估模型医学问答能力，默认处理 10 条样本用于快速验证。

```bash
cd examples/biosciences/medgemma
bash scripts/run_evaluate_on_medqa.sh
```

脚本内部调用：

```bash
python ./notebook_conver/evaluate_on_medqa.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --parquet_dir ${ONESCIENCE_DATASETS_DIR}/medgemma/medqa \
    --output_dir ./medqa_results \
    --max_samples 10
```

常用参数覆盖（通过修改脚本或环境变量）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it` | 模型目录 |
| `--parquet_dir` | `${ONESCIENCE_DATASETS_DIR}/medgemma/medqa` | MedQA parquet 数据目录 |
| `--output_dir` | `./medqa_results` | 结果输出目录 |
| `--max_samples` | `10` | 评估样本数，设置为 `-1` 可评估全部 |
| `--device` / `HIP_VISIBLE_DEVICES` | `0` | GPU 设备 |

输出：

- `scripts/medqa_results/medqa_results.json`：每条样本的详细结果
- `scripts/medqa_results/summary.txt`：准确率等汇总指标

---

### 3. 胸片解剖结构定位（`run_cxr_anatomy.sh`）

对单张或多张胸片进行解剖部位定位。脚本内部同时运行单图模式和批量模式：

```bash
cd examples/biosciences/medgemma
bash scripts/run_cxr_anatomy.sh
```

脚本内部调用：

```bash
# 单图模式
python ./notebook_conver/cxr_anatomy_localization_with_hugging_face.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --image_path "${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/covid/COVID-19 (89).jpg" \
    --object_name "right clavicle" \
    --num_gpus 2

# 多图模式
python ./notebook_conver/cxr_anatomy_localization_with_hugging_face.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --input_dir "${ONESCIENCE_DATASETS_DIR}/medgemma/test_images" \
    --object_name "right clavicle" \
    --num_gpus 2
```

常用参数：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--model_path` | 是 | 本地模型目录 |
| `--image_path` | 单图模式必填 | 单张胸片路径 |
| `--input_dir` | 批量模式必填 | 批量胸片目录 |
| `--object_name` | 是 | 待定位的解剖结构，例如 `"right clavicle"` |
| `--num_gpus` | 否 | 使用的 GPU 数量，默认 `1` |
| `--output_dir` | 否 | 结果输出目录，默认 `./outputs` |

输出：

- `scripts/outputs/result_*.json`：定位坐标与标签
- `scripts/outputs/result_*.png`：带边界框标注的可视化图像
- `scripts/outputs/batch_summary.json`：批量模式汇总结果

---

### 4. 胸片纵向对比分析（`run_cxr_longitudinal_comparison.sh`）

对同一患者的前后两次胸片进行对比分析：

```bash
cd examples/biosciences/medgemma
bash scripts/run_cxr_longitudinal_comparison.sh
```

脚本内部调用：

```bash
python ./notebook_conver/cxr_longitudinal_comparison.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --image1 ${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/longitudinal_cxr_before.png \
    --image2 ${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/longitudinal_cxr_after.png \
    --output_dir ./compare_outputs
```

常用参数：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--model_path` | 是 | 本地模型目录 |
| `--image1` | 是 | 第一张图像路径（如治疗前） |
| `--image2` | 是 | 第二张图像路径（如治疗后） |
| `--output_dir` | 否 | 结果输出目录，默认 `./compare_outputs` |
| `--prompt` | 否 | 自定义对比提示词 |
| `--preprocess` | 否 | 图像非正方形时是否填充为正方形 |
| `--num_gpus` | 否 | 使用的 GPU 数量 |

输出：

- `scripts/compare_outputs/compare_<image1>_vs_<image2>.txt`：文本对比报告
- `scripts/compare_outputs/compare_<image1>_vs_<image2>.json`：结构化 JSON 结果

---

### 5. 病理图像 LoRA 微调（`run_fine_tune.sh`）

基于 NCT 结肠组织病理图像数据集进行 LoRA 微调：

```bash
cd examples/biosciences/medgemma
bash scripts/run_fine_tune.sh
```

脚本内部调用：

```bash
python ./notebook_conver/fine_tune_with_hugging_face.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --train_zip ${ONESCIENCE_DATASETS_DIR}/medgemma/nct/NCT-CRC-HE-100K.zip \
    --test_zip ${ONESCIENCE_DATASETS_DIR}/medgemma/nct/CRC-VAL-HE-7K.zip \
    --output_dir ./medgemma-nct-lora \
    --max_train_samples 9000 \
    --max_val_samples 1000 \
    --max_test_samples 1000
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it` | 本地模型目录 |
| `--train_zip` | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/NCT-CRC-HE-100K.zip` | 训练集 zip |
| `--test_zip` | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/CRC-VAL-HE-7K.zip` | 测试集 zip |
| `--output_dir` | `./medgemma-nct-lora` | LoRA 输出目录 |
| `--max_train_samples` | `9000` | 训练样本数 |
| `--max_val_samples` | `1000` | 验证样本数 |
| `--max_test_samples` | `1000` | 测试样本数 |

输出：

- `scripts/medgemma-nct-lora/`：LoRA 权重、训练日志与评估结果

> 注：脚本会自动检查并修复 `boto3==1.43.36` 和 `botocore==1.43.36` 版本，避免依赖冲突。

---

### 6. 使用推理运行器

`runner/medical_inference_runner.py` 提供统一的推理入口，支持交互式与批量文件推理。

#### 交互式推理

```bash
cd examples/biosciences/medgemma
export PYTHONPATH=../../../src:$PYTHONPATH
python runner/medical_inference_runner.py \
    --config configs/inference_config.yaml \
    --interactive
```

#### 批量文件推理

```bash
cd examples/biosciences/medgemma
export PYTHONPATH=../../../src:$PYTHONPATH
python runner/medical_inference_runner.py \
    --config configs/inference_config.yaml \
    --input data/example_input.json
```

---

### 7. Python API 调用

```python
from onescience.models.medgemma import MedGemma
from onescience.models.medgemma.config import load_config

configs = load_config("configs/inference_config.yaml")
model = MedGemma(configs)

messages = [
    {"role": "system", "content": "You are an expert medical AI assistant."},
    {"role": "user", "content": "What are the common causes of hypertension?"}
]
result = model.forward(messages, max_tokens=500)
print(result["choices"][0]["message"]["content"])
```

---

## 目录结构

```
examples/biosciences/medgemma/
├── configs/                          # 配置目录
│   ├── inference_config.yaml         # 推理配置示例
│   └── configs_base.py               # 基础配置定义
├── runner/
│   └── medical_inference_runner.py   # 统一医学推理运行器
├── scripts/                          # 可执行脚本
│   ├── notebook_conver/              # 脚本调用的 Python 实现
│   │   ├── cxr_anatomy_localization_with_hugging_face.py
│   │   ├── cxr_longitudinal_comparison.py
│   │   ├── evaluate_on_medqa.py
│   │   ├── fine_tune_with_hugging_face.py
│   │   └── detect_image_token.py
│   ├── run_cxr_anatomy.sh            # 胸片解剖结构定位（已提供）
│   ├── run_cxr_longitudinal_comparison.sh  # 胸片前后对比分析（已提供）
│   ├── run_evaluate_on_medqa.sh      # MedQA 医学问答评估（已提供）
│   └── run_fine_tune.sh              # 病理图像 LoRA 微调（已提供）
├── data/                             # 示例输入数据
│   ├── example_input.json
│   └── train_example.jsonl
├── tests/
│   └── test_integration.py           # 集成测试脚本
└── READMD.md                         # 本文档
```

模型实现位于 `src/onescience/models/medgemma`。

---

## 注意事项

- 运行脚本前需确保 `ONESCIENCE_DATASETS_DIR` 环境变量已正确设置。
- 脚本默认使用 `HIP_VISIBLE_DEVICES=0`，在海光 DCU 平台可直接运行；在 CUDA 平台可替换为 `CUDA_VISIBLE_DEVICES=0` 或根据设备调整。
- 如需使用 vLLM 加速推理，请确保已安装对应版本的 vLLM 并配置 `use_vllm: true`。
- 胸片解剖定位与病理微调脚本会自动修复 `boto3` / `botocore` 版本，避免依赖冲突。
- 所有 sh 脚本内部自动加载项目根目录 `env.sh`，无需手动 source。
- 4B 多模态模型推理显存需求较大，建议至少单卡 24GB 显存；多卡可通过 `num_gpus` 或外部 `CUDA_VISIBLE_DEVICES` 控制。

---

## 许可证与引用

MedGemma 模型采用 [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms) 许可，本仓库示例代码采用 Apache 2.0 许可。

更多信息请参阅：

- [开发者文档](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [模型卡](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [社区准则](https://developers.google.com/health-ai-developer-foundations/community-guidelines)
- [Hugging Face](https://huggingface.co/models?other=medgemma)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)
