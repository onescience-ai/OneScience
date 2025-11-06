#!/usr/bin/env python3
import os
import argparse
import torch
import pickle  # 导入 pickle 模块，这是新方法的核心
from omegaconf import OmegaConf, DictConfig, ListConfig

# --- “特殊方法”核心：使用自定义 Unpickler 在加载时替换模块路径 ---
# 1. 定义路径的映射规则
PATH_MAP = {
    # 规则一：更具体的路径优先替换，防止被通用规则覆盖
    "fairchem.core.models": "onescience.models.UMA.models",
    # 规则二：更通用的路径在后
    "fairchem.core": "onescience.models.UMA"
}

# 2. 创建一个继承自 pickle.Unpickler 的自定义类
class CustomUnpickler(pickle.Unpickler):
    """
    这是一个自定义的解包器。
    它重写了 find_class 方法，在加载时动态地重定向模块路径。
    """
    def find_class(self, module_name, class_name):
        redirected_module_name = module_name
        for old_path, new_path in PATH_MAP.items():
            if module_name.startswith(old_path):
                redirected_module_name = module_name.replace(old_path, new_path, 1)
                # 找到第一个匹配的规则后即可停止
                break
        
        # 调用父类的 find_class 方法，但传入的是我们可能修改过的 module_name
        return super().find_class(redirected_module_name, class_name)
# --- 特殊方法结束 ---


def export_from_infer_pt(infer_pt: str):
    """从 .pt 文件中加载并提取所需内容。"""
    # 为了将我们的 CustomUnpickler 注入到 torch.load 中，
    # 我们创建一个模拟的 pickle 模块。
    class CustomPickleModule:
        Unpickler = CustomUnpickler

    # 现在调用 torch.load，它会使用我们提供的 Unpickler 来查找类路径，
    # 同时保留其处理张量等复杂数据的能力 (persistent_load)。
    obj = torch.load(infer_pt, map_location="cpu", pickle_module=CustomPickleModule)

    if hasattr(obj, "model_state_dict"):
        return {
            "model_sd":  obj.model_state_dict,
            "ema_sd":    getattr(obj, "ema_state_dict", None),
            "model_cfg": obj.model_config,
            "tasks_cfg": obj.tasks_config,
        }
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return {
            "model_sd":  obj["model_state_dict"],
            "ema_sd":    obj.get("ema_state_dict"),
            "model_cfg": obj["model_config"],
            "tasks_cfg": obj["tasks_config"],
        }
    raise RuntimeError("输入文件似乎不是一个有效的 MLIPInferenceCheckpoint 结构。")

def to_jsonable(x):
    """将 OmegaConf 对象转换为纯 Python 对象。"""
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=True, enum_to_str=True)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x

def recursive_replace_paths(obj, old_prefix, new_prefix):
    """
    递归遍历数据结构，替换配置字典中以字符串形式存在的路径。
    """
    if isinstance(obj, str):
        if obj.startswith(old_prefix):
            return obj.replace(old_prefix, new_prefix, 1)
        return obj
    if isinstance(obj, dict):
        return {k: recursive_replace_paths(v, old_prefix, new_prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_replace_paths(v, old_prefix, new_prefix) for v in obj]
    return obj

def main():
    ap = argparse.ArgumentParser(
        description="[高级版] 将 UMA 模型的 .pt 文件从 'fairchem.core' 命名空间转换为 'onescience.models.UMA'，无需安装 fairchem。"
    )
    ap.add_argument("src", help="原始的 .pt 模型文件路径（例如 uma-m-1p1.pt）")
    ap.add_argument("--out", required=True, help="转换后输出的 .pt 新文件路径")
    args = ap.parse_args()

    try:
        print(f"[*] 正在使用自定义Unpickler加载原始模型: {args.src}")
        # 1. 加载模型
        bundle = export_from_infer_pt(args.src)

        # 2. 将配置转换为可修改的 Python 字典
        model_cfg = to_jsonable(bundle["model_cfg"])
        tasks_cfg = to_jsonable(bundle["tasks_cfg"])

        # 3. 在内存中替换配置字典内部的字符串路径
        print("[*] 正在转换配置字典内的模块路径...")
        new_model_cfg = recursive_replace_paths(
            model_cfg, "fairchem.core.models", "onescience.models.UMA.models"
        )
        print("    - 模型配置路径已更新。")
        new_tasks_cfg = recursive_replace_paths(
            tasks_cfg, "fairchem.core", "onescience.models.UMA"
        )
        print("    - 任务配置路径已更新。")

        # 4. 重新构建并保存为新模型
        print("[*] 正在构建新的模型检查点对象...")
        from onescience.models.UMA.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
        
        converted_checkpoint = MLIPInferenceCheckpoint(
            model_state_dict=bundle["model_sd"],
            ema_state_dict=(bundle.get("ema_sd") or None),
            model_config=new_model_cfg,
            tasks_config=new_tasks_cfg,
        )

        print(f"[*] 正在保存转换后的模型到: {args.out}")
        torch.save(converted_checkpoint, args.out)

        print("\n[成功] 模型转换完成！")

    except ImportError:
        print("\n[错误!] 无法从 'onescience.models.UMA' 中导入 MLIPInferenceCheckpoint。")
        print("请确保您的 'onescience' 包已正确安装在当前 Python 环境中。")
    except Exception as e:
        print(f"\n[错误] 操作过程中出现问题: {e}")

if __name__ == "__main__":
    main()

