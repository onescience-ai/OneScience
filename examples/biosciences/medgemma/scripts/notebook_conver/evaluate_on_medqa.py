#!/usr/bin/env python3
# medqa_eval.py – MedGemma on MedQA (仿照 c6.py 风格)
# 功能：
#   1. 加载本地 MedGemma 模型（纯文本推理）
#   2. 加载 MedQA 数据集（支持在线或本地 Parquet）
#   3. 逐条生成答案并提取选项
#   4. 计算准确率，保存详细结果和汇总

import os
import sys
import argparse
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import datasets

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


def format_prompt(question: str, options: dict) -> str:
    """构造选择题提示词"""
    options_str = f"(A) {options['A']} (B) {options['B']} (C) {options['C']} (D) {options['D']}"
    prompt = f"""Answer the given question. Think step by step.
You can directly provide the answer (A single letter), without further additions. E.g. "Final Answer: (A)".
Question: {question}
{options_str}
"""
    return prompt


# 答案提取正则（与原脚本一致）
ANSWER_PATTERNS = [
    r'The final answer is\s\(([A-J])\)',
    r'The final answer is\s\**\(([A-J])\)\**',
    r'The final answer is\s\$\\boxed{([A-J])}\$',
    r'Final Answer:\(([A-J])\)',
    r'Final Answer:\s\(([A-J])\)',
    r'Final Answer:\s\(?([A-J])',
    r'Final Answer:\s*\**\(([A-J])\)\**',
    r'\**Final Answer:\**\s\(([A-J])\)',
]


#def extract_answer(text: str) -> str:
#    """从模型回复中提取选项字母"""
#    if not isinstance(text, str) or not text:
#        return None
#    for pat in ANSWER_PATTERNS:
#        m = re.search(pat, text)
#        if m:
#            return m.group(1)
#    return None

def extract_answer(text: str) -> str:
    """从模型回复中提取选项字母，支持多种回退策略"""
    if not isinstance(text, str) or not text:
        return None

    # 1. 优先用原始正则匹配标准格式
    for pat in ANSWER_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    # 2. 回退：在 <unused95> 之后（正式回答部分）寻找最后一个单独的大写字母
    if "<unused95>" in text:
        post_think = text.split("<unused95>")[-1]
        # 寻找形如 " (A)" 或 "(A)" 的选项
        matches = re.findall(r'\(([A-J])\)', post_think)
        if matches:
            return matches[-1]   # 通常最后一个选项是最终答案
        # 如果仍未找到，尝试寻找单独的大写字母（可能模型说 "Answer: A"）
        m = re.search(r'\b([A-J])\b\s*$', post_think)
        if m:
            return m.group(1)

    # 3. 全局回退：在整个回复中找最后出现的 (X) 格式
    matches = re.findall(r'\(([A-J])\)', text)
    if matches:
        return matches[-1]

    return None


def main():
    parser = argparse.ArgumentParser(description="MedGemma MedQA 评测")
    parser.add_argument("--model_path", required=True, help="本地 MedGemma 模型路径")
    parser.add_argument("--output_dir", default="./medqa_outputs", help="输出目录（保存结果）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制评测样本数（默认全部）")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="使用的 GPU 数量（建议在命令行用 CUDA_VISIBLE_DEVICES 控制）")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度（0 为确定性）")
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="本地 MedQA Parquet 文件夹路径（包含 train/test/dev 等文件）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # GPU 设置提示
    if args.num_gpus:
        print("提示：请通过环境变量限制 GPU，例如 CUDA_VISIBLE_DEVICES=0,1")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")

    # 加载本地模型（纯文本推理，仍使用 AutoModelForImageTextToText）
    print(f"Loading model from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True
    )
    print("Model loaded.\n")

    # 加载 MedQA 数据集
    if args.parquet_dir:
        print(f"Loading MedQA dataset from local Parquet directory: {args.parquet_dir}")
        data_files = {
            "train": os.path.join(args.parquet_dir, "train-*.parquet"),
            "test": os.path.join(args.parquet_dir, "test-*.parquet"),
            "validation": os.path.join(args.parquet_dir, "dev-*.parquet"),  # 验证集文件前缀为 dev
        }
        dataset = datasets.load_dataset("parquet", data_files=data_files)
    else:
        print("Loading MedQA dataset from openlifescienceai/medqa...")
        dataset = datasets.load_dataset("openlifescienceai/medqa")

    test_data = dataset["test"]
    if args.max_samples and args.max_samples < len(test_data):
        test_data = test_data.select(range(args.max_samples))
    print(f"Total test samples: {len(test_data)}")

    # 准备结果容器
    results = []
    correct = 0

    # 逐条推理
    for idx, item in enumerate(tqdm(test_data, desc="Inference")):
        data = item["data"]  # 注意数据集结构：item["data"] 包含 Question, Options, Correct Option
        question = data["Question"]
        options = data["Options"]
        gold_answer = data["Correct Option"]

        prompt = format_prompt(question, options)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "SYSTEM INSTRUCTION: think silently if needed."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        # 应用聊天模板
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None
            )

        input_len = inputs["input_ids"].shape[1]
        response = processor.batch_decode(gen[:, input_len:], skip_special_tokens=True)[0]

        # 提取答案
        extracted = extract_answer(response)
        is_correct = (extracted == gold_answer)
        if is_correct:
            correct += 1

        # 保存单条结果
        results.append({
            "index": idx,
            "question": question,
            "options": options,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": extracted,
            "correct": is_correct
        })

    # 计算准确率
    total = len(test_data)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")

    # 保存结果
    output_file = os.path.join(args.output_dir, "medqa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "total": total, "correct": correct, "details": results}, f, indent=2)
    print(f"Results saved to {output_file}")

    # 同时输出摘要文本
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
