#!/usr/bin/env python3
# c6.py - MedGemma 胸部X光解剖结构定位（增强版）
# 功能：
#   - 单张/批量图片推理
#   - 指定输出目录
#   - 指定使用的GPU数量
#   - 坐标自动适配（0-1 或 0-1000）
#   - 本地模型加载（local_files_only）

import os
import sys
import argparse
import json
import re
import torch
import numpy as np
import skimage
from PIL import Image, ImageDraw
from pathlib import Path
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


def draw_bounding_box(image, bbox_data, object_name):
    """绘制边界框（自动识别 0-1 或 0-1000 坐标）"""
    w_orig, h_orig = image.size
    new_w = 512
    new_h = int(h_orig * (new_w / w_orig))
    scaled = image.resize((new_w, new_h))
    draw = ImageDraw.Draw(scaled)

    for item in bbox_data:
        box = item.get("box_2d")
        label = item.get("label", object_name)
        if box and len(box) == 4:
            y0, x0, y1, x1 = box

            # 若坐标值均在 [-1,1] 区间（实际上≤1），视为 0-1 归一化，乘以 1000
            if max(abs(y0), abs(x0), abs(y1), abs(x1)) <= 1.0:
                y0, x0, y1, x1 = y0 * 1000, x0 * 1000, y1 * 1000, x1 * 1000

            # 转换到像素坐标（以 512 宽为基准）
            x0_px = x0 / 1000 * new_w
            y0_px = y0 / 1000 * new_h
            x1_px = x1 / 1000 * new_w
            y1_px = y1 / 1000 * new_h

            draw.rectangle([(x0_px, y0_px), (x1_px, y1_px)], outline="red", width=2)
            if label:
                draw.text((x0_px, y0_px - 15), label, fill="red")
    return scaled


def build_prompt(object_name):
    """构造单结构定位提示词"""
    return f"""Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""


def extract_bbox_data(response):
    """从模型回复中提取 JSON 边界框列表（增强鲁棒性）"""
    json_str = ""
    # 策略1：优先提取 Final Answer: 后面的 ```json...```
    final_match = re.search(r'Final Answer:\s*```json\s*(\[.*?\])\s*```', response, re.DOTALL)
    if final_match:
        json_str = final_match.group(1).strip()
    else:
        # 策略2：查找 ```json...```
        if "```json" in response:
            start = response.find("```json") + len("```json")
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
    if not json_str:
        # 回退：提取最后一个 JSON 数组
        matches = re.findall(r'\[.*\]', response, re.DOTALL)
        if matches:
            json_str = matches[-1]

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"    JSON 解析错误: {e}\n    提取的字符串: {json_str}")
    return []


def process_single_image(img_path, model, processor, object_name, output_dir, preprocess=True):
    """处理单张图像，返回结果字典"""
    print(f"\n{'='*60}\nProcessing: {img_path}")
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size
    if preprocess:
        image = Image.fromarray(pad_image_to_square(np.array(image)))
        print(f"  Preprocessed: {image.size} (original: {orig_size})")

    prompt = build_prompt(object_name)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]

    # 处理输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(gen[:, input_len:], skip_special_tokens=True)[0]

    # 提取坐标
    bbox_data = extract_bbox_data(response)
    print(f"  Detected {len(bbox_data)} bounding box(es) for '{object_name}'.")

    # 保存标注图片
    base = Path(img_path).stem
    out_img_path = os.path.join(output_dir, f"result_{base}_{object_name.replace(' ', '_')}.png")
    if bbox_data:
        draw_bounding_box(image, bbox_data, object_name).save(out_img_path)
        print(f"  Annotated image saved: {out_img_path}")

    result = {
        "image_path": str(img_path),
        "original_size": list(orig_size),
        "object_name": object_name,
        "bounding_boxes": bbox_data,
        "raw_response": response
    }
    # 保存单张 JSON（可选）
    json_path = os.path.join(output_dir, f"result_{base}_{object_name.replace(' ', '_')}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="MedGemma 胸部X光单结构定位（支持批量）")
    parser.add_argument("--image_path", help="单张图像路径")
    parser.add_argument("--input_dir", help="批量处理图像文件夹")
    parser.add_argument("--model_path", required=True, help="本地模型目录路径")
    parser.add_argument("--object_name", default="right clavicle", help="要定位的解剖结构名称")
    parser.add_argument("--output_dir", default="./outputs", help="结果保存目录（默认 ./outputs）")
    parser.add_argument("--num_gpus", type=int, default=None, help="使用的GPU数量（如 2）")
    parser.add_argument("--preprocess", action="store_true", default=True, help="是否将图像填充为正方形")
    args = parser.parse_args()

    if not args.image_path and not args.input_dir:
        parser.error("必须指定 --image_path 或 --input_dir 之一")
    if args.image_path and args.input_dir:
        parser.error("不能同时指定 --image_path 和 --input_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    # GPU 设置
    if args.num_gpus and args.num_gpus > 0:
        available = torch.cuda.device_count()
        num = min(args.num_gpus, available)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num))
        print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} (available: {available})")
    else:
        print(f"All GPUs ({torch.cuda.device_count()}) available.")

    # 加载模型
    print(f"Loading model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
        #local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
        #local_files_only=True
    )
    print("Model loaded.\n")

    # 构建图像列表
    if args.image_path:
        image_paths = [args.image_path]
    else:
        # 扫描文件夹中所有常见图像文件
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_paths = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(exts)
        ])
        if not image_paths:
            print(f"No images found in {args.input_dir}")
            return
    print(f"Images to process: {len(image_paths)}, target: '{args.object_name}'")

    # 逐张处理
    all_results = []
    for p in image_paths:
        try:
            res = process_single_image(p, model, processor, args.object_name, args.output_dir, args.preprocess)
            all_results.append(res)
        except Exception as e:
            print(f"  Error processing {p}: {e}")

    # 批量汇总
    if len(image_paths) > 1 or args.input_dir:
        summary_path = os.path.join(args.output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBatch summary saved to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
