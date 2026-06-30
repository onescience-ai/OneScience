#!/usr/bin/env python3
# train_nct.py – MedGemma 组织分类微调 & 评估 (仿照 c6.py)
# 功能：
#   1. 加载本地 NCT-CRC-HE-100K 训练集 和 CRC-VAL-HE-7K 测试集（自动解压 zip）
#   2. 使用 QLoRA 微调 MedGemma（4-bit 量化）
#   3. 在测试集上评估准确率和 F1

import os
import sys
import argparse
import json
import zipfile
import torch
import numpy as np
from typing import Any
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
#    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import evaluate

# ---------- Monkey-patch for torch<2.6 ----------
import transformers.masking_utils as masking_utils
if torch.__version__ < "2.6":
    _orig_causal = masking_utils.create_causal_mask
    _orig_sliding = masking_utils.create_sliding_window_causal_mask

    def _safe_pop(kw):
        kw.pop("or_mask_function", None)
        kw.pop("and_mask_function", None)

    def _new_causal(*a, **kw):
        _safe_pop(kw)
        return _orig_causal(*a, **kw)

    def _new_sliding(*a, **kw):
        _safe_pop(kw)
        return _orig_sliding(*a, **kw)

    masking_utils.create_causal_mask = _new_causal
    masking_utils.create_sliding_window_causal_mask = _new_sliding


TISSUE_CLASSES = [
    "A: adipose",
    "B: background",
    "C: debris",
    "D: lymphocytes",
    "E: mucus",
    "F: smooth muscle",
    "G: normal colon mucosa",
    "H: cancer-associated stroma",
    "I: colorectal adenocarcinoma epithelium",
]

PROMPT = f"What is the most likely tissue type shown in the histopathology image?\n" + "\n".join(TISSUE_CLASSES)


def extract_zip(zip_path: str, extract_to: str) -> str:
    """解压 zip 文件，返回解压后的目录路径"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    return extract_to


def format_data(example: dict[str, Any]) -> dict[str, Any]:
    """构造训练 messages 格式"""
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": TISSUE_CLASSES[example["label"]]},
            ],
        },
    ]
    return {
        "image": example["image"],   # 必须包含这一行
        "text": "...处理好的文本..."  # 必须有文本字段
    }
    #return example


def format_test_data(example: dict[str, Any]) -> dict[str, Any]:
    """构造测试 messages 格式（无 assistant）"""
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    return example


def collate_fn(examples: list[dict[str, Any]], processor):
    texts = []
    images = []
    for example in examples:
        images.append([example["image"].convert("RGB")])
        texts.append(
            processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            ).strip()
        )

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    # 掩码图像 token 和填充 token
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for tok_id in image_token_id:
        labels[labels == tok_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch


def postprocess(prediction: list[dict[str, str]], do_full_match: bool = False) -> int:
    """将模型预测转换为类别索引"""
    response_text = prediction[0]["generated_text"]
    if do_full_match:
        try:
            return LABEL_FEATURE.str2int(response_text)
        except:
            return -1
    for label in TISSUE_CLASSES:
        if label in response_text or f"({label.replace(': ', ') ')}" in response_text:
            return LABEL_FEATURE.str2int(label)
    return -1


def main():
    parser = argparse.ArgumentParser(description="MedGemma NCT 组织分类微调与评估")
    parser.add_argument("--model_path", required=True, help="本地 MedGemma 模型路径")
    parser.add_argument("--train_zip", default="./NCT-CRC-HE-100K.zip", help="训练集 zip 路径")
    parser.add_argument("--test_zip", default="./CRC-VAL-HE-7K.zip", help="测试集 zip 路径")
    parser.add_argument("--output_dir", default="./medgemma-nct-lora", help="微调模型输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=9000, help="训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=1000, help="验证样本数")
    parser.add_argument("--max_test_samples", type=int, default=1000, help="测试样本数（评估用）")
    parser.add_argument("--num_gpus", type=int, default=None, help="使用的 GPU 数量（建议在命令行用 CUDA_VISIBLE_DEVICES 控制）")
    parser.add_argument("--eval_only", action="store_true", help="仅评估，跳过训练")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练（与 eval_only 相同）")
    args = parser.parse_args()

    # GPU 设置提示
    if args.num_gpus:
        print("提示：请通过环境变量限制 GPU，例如 CUDA_VISIBLE_DEVICES=0,1")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")

    # 解压数据集
    train_dir = os.path.splitext(args.train_zip)[0]
    test_dir = os.path.splitext(args.test_zip)[0]
    if not os.path.exists(train_dir):
        print(f"解压训练集 {args.train_zip} -> {train_dir}")
        extract_zip(args.train_zip, train_dir)
    if not os.path.exists(test_dir):
        print(f"解压测试集 {args.test_zip} -> {test_dir}")
        extract_zip(args.test_zip, test_dir)

    # 加载数据集
    print("加载训练集...")
    data = load_dataset(train_dir, split="train")
    data = data.train_test_split(
        train_size=args.max_train_samples,
        test_size=args.max_val_samples,
        shuffle=True,
        seed=42,
    )
    data["validation"] = data.pop("test")

    # 格式化数据
    #data = data.map(format_data, remove_columns=data["train"].column_names)
    data = data.map(format_data)

    # 加载模型 (QLoRA 量化)
    print(f"加载模型 {args.model_path}...")
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
   # model_kwargs["quantization_config"] = BitsAndBytesConfig(
   #     load_in_4bit=True,
   #     bnb_4bit_use_double_quant=True,
   #     bnb_4bit_quant_type="nf4",
   #     bnb_4bit_compute_dtype=torch.bfloat16,
   #     bnb_4bit_quant_storage=torch.bfloat16,
   # )

    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"  # 训练用右填充

    # LoRA 配置
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # 训练参数
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=50,
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    if not args.eval_only and not args.skip_train:
        print("开始训练...")
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=data["train"],
            eval_dataset=data["validation"].select(range(min(200, len(data["validation"])))),
            peft_config=peft_config,
            processing_class=processor,
            data_collator=lambda examples: collate_fn(examples, processor),
        )
        trainer.train()
        trainer.save_model()
        print(f"模型已保存到 {args.output_dir}")
        del trainer
        torch.cuda.empty_cache()

    # ---------- 评估 ----------
    print("加载测试集...")
    test_data = load_dataset(test_dir, split="train")
    test_data = test_data.shuffle(seed=42).select(range(args.max_test_samples))
    test_data = test_data.map(format_test_data, remove_columns=test_data.column_names)

    # 设置标签特征（全局变量，供 postprocess 使用）
    global LABEL_FEATURE
    test_data = test_data.cast_column("label", ClassLabel(names=TISSUE_CLASSES))
    LABEL_FEATURE = test_data.features["label"]

    # 构建评估管道
    model_id = args.output_dir if (args.eval_only or args.skip_train) else args.model_path

    print(f"创建评估管道，模型: {model_id}")
    eval_pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        processor=processor,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    eval_pipe.model.generation_config.do_sample = False
    eval_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"  # 推理用左填充

    # 执行推理
    print("运行推理...")
    outputs = eval_pipe(
        text=test_data["messages"],
        images=test_data["image"],
        max_new_tokens=40,
        batch_size=64,
        return_full_text=False,
    )

    # 后处理
    do_full_match = args.eval_only or args.skip_train  # 微调后模型输出可能更精确
    predictions = [postprocess(out, do_full_match) for out in outputs]
    references = test_data["label"]

    # 计算指标
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    acc = accuracy_metric.compute(predictions=predictions, references=references)
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")
    metrics = {**acc, **f1}
    print(f"评估结果: {metrics}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, "eval_metrics.json")
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"结果已保存到 {result_file}")


if __name__ == "__main__":
    main()
