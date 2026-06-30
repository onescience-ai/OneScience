#!/usr/bin/env python3
# compare_cxr.py - MedGemma 纵向CXR对比（仿照 c6.py）
# 功能：
#   - 比较两张胸部X光片（如治疗前后），生成描述报告
#   - 支持本地模型、指定GPU数量、自定义prompt、图像预处理
#   - 结果保存为文本文件

import os
import sys
import argparse
import json
import torch
import numpy as np
import skimage
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

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


def pad_image_to_square(image_array):
    """将图像填充为正方形（与训练预处理一致）"""
    image_array = skimage.util.img_as_ubyte(image_array)
    if image_array.ndim < 3:
        image_array = skimage.color.gray2rgb(image_array)
    if image_array.shape[2] == 4:
        image_array = skimage.color.rgba2rgb(image_array)

    h, w = image_array.shape[:2]
    if h < w:
        dh = w - h
        image_array = np.pad(image_array, ((dh // 2, dh - dh // 2), (0, 0), (0, 0)))
    elif w < h:
        dw = h - w
        image_array = np.pad(image_array, ((0, 0), (dw // 2, dw - dw // 2), (0, 0)))
    return image_array


def build_default_prompt():
    """默认的纵向CXR对比提示词"""
    return (
        "Provide a comparison of these two images and include details from "
        "the image which students should take note of when reading longitudinal CXR."
    )


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma 纵向胸部X光片对比"
    )
    parser.add_argument("--model_path", required=True, help="本地模型目录路径")
    parser.add_argument("--image1", required=True, help="第一张图像路径（如治疗前）")
    parser.add_argument("--image2", required=True, help="第二张图像路径（如治疗后）")
    parser.add_argument("--prompt", default=None,
                        help="自定义对比提示词（不指定则使用默认）")
    parser.add_argument("--output_dir", default="./outputs",
                        help="结果保存目录（默认 ./outputs）")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="使用的GPU数量（建议在命令行用 CUDA_VISIBLE_DEVICES 控制）")
    parser.add_argument("--preprocess", action="store_true", default=False,
                        help="是否将图像填充为正方形（默认不填充）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # GPU 设置（若在脚本内设，需在 import torch 之前，此处仅作提示）
    if args.num_gpus and args.num_gpus > 0:
        # 实际限制应在运行脚本前通过环境变量设置，这里只打印建议
        print(f"提示：建议在命令行中设置 CUDA_VISIBLE_DEVICES=0,1,... 来限制GPU。"
              f"当前脚本不修改可见设备。")
    else:
        print(f"可用GPU数量: {torch.cuda.device_count()}")

    # 加载模型（全本地）
    print(f"Loading model from {args.model_path}")
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

    # 加载图像
    print(f"Loading images:\n  {args.image1}\n  {args.image2}")
    image1 = Image.open(args.image1).convert("RGB")
    image2 = Image.open(args.image2).convert("RGB")
    print(f"Original sizes: {image1.size}, {image2.size}")

    if args.preprocess:
        img1_arr = pad_image_to_square(np.array(image1))
        img2_arr = pad_image_to_square(np.array(image2))
        image1 = Image.fromarray(img1_arr)
        image2 = Image.fromarray(img2_arr)
        print(f"Preprocessed sizes: {image1.size}, {image2.size}")

    # 构造提示词
    prompt = args.prompt if args.prompt else build_default_prompt()
    print(f"Prompt: {prompt[:100]}...")

    # 构造 messages（包含两张图像）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # 处理输入
    print("Processing inputs...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")

    # 推理
    print("Running inference...")
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        generate_ids[:, input_len:], skip_special_tokens=True
    )[0]

    # 输出结果
    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print(response)
    print("=" * 60)

    # 保存为文本文件
    base1 = os.path.splitext(os.path.basename(args.image1))[0]
    base2 = os.path.splitext(os.path.basename(args.image2))[0]
    out_txt = os.path.join(args.output_dir, f"compare_{base1}_vs_{base2}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Image1: {args.image1}\nImage2: {args.image2}\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Response:\n{response}\n")
    print(f"Comparison result saved to: {out_txt}")

    # 可选保存 JSON
    out_json = os.path.join(args.output_dir, f"compare_{base1}_vs_{base2}.json")
    result = {
        "image1": args.image1,
        "image2": args.image2,
        "prompt": prompt,
        "response": response
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
