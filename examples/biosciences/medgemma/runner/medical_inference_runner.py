# MedGemma 医学推理运行器
# 类似 Protenix 的 BiologyInferenceRunner

import logging
import os
import json
from typing import Any, Dict, List, Optional
import torch

from onescience.models.medgemma.medgemma import MedGemma
from onescience.models.medgemma.config import parse_configs, load_config

logger = logging.getLogger(__name__)


class MedicalInferenceRunner:
    """
    MedGemma 医学推理运行器
    提供统一的推理接口
    """

    def __init__(self, configs: Any) -> None:
        """
        初始化推理运行器

        Args:
            configs: 配置对象
        """
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.init_dumper()

        logger.info("MedicalInferenceRunner initialized")

    def init_env(self) -> None:
        """初始化环境"""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

    def init_basics(self) -> None:
        """初始化基础设置"""
        self.dump_dir = self.configs.output.dump_dir
        self.error_dir = os.path.join(self.dump_dir, "errors")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        logger.info(f"Output directory: {self.dump_dir}")

    def init_model(self) -> None:
        """初始化 MedGemma 模型"""
        logger.info("Loading MedGemma model...")
        try:
            self.model = MedGemma(self.configs)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def init_dumper(self) -> None:
        """初始化结果保存器"""
        self.output_format = self.configs.output.output_format
        self.save_predictions = self.configs.output.save_predictions
        logger.info(f"Output format: {self.output_format}")

    @torch.no_grad()
    def predict(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        运行推理

        Args:
            messages: 消息列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            预测结果
        """
        if max_tokens is None:
            max_tokens = self.configs.inference.default_max_tokens
        if temperature is None:
            temperature = self.configs.inference.temperature

        try:
            result = self.model.forward(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def run_from_file(self, input_path: str) -> None:
        """
        从文件运行推理

        Args:
            input_path: 输入文件路径（JSON 或 JSONL）
        """
        logger.info(f"Loading input from: {input_path}")

        # 读取输入数据
        if input_path.endswith('.jsonl'):
            samples = self._load_jsonl(input_path)
        elif input_path.endswith('.json'):
            samples = self._load_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

        logger.info(f"Loaded {len(samples)} samples")

        # 处理每个样本
        results = []
        for idx, sample in enumerate(samples):
            logger.info(f"Processing sample {idx + 1}/{len(samples)}")

            try:
                # 提取消息
                if "messages" in sample:
                    messages = sample["messages"]
                elif "text" in sample:
                    messages = [{"role": "user", "content": sample["text"]}]
                elif "question" in sample:
                    messages = [{"role": "user", "content": sample["question"]}]
                else:
                    logger.warning(f"Sample {idx} has no valid input")
                    continue

                # 运行推理
                result = self.predict(messages)

                # 添加样本 ID
                result["sample_id"] = sample.get("id", idx)

                # 保存结果
                if self.save_predictions:
                    self._save_result(result, idx)

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                self._save_error(sample, idx, str(e))

        logger.info(f"Completed processing {len(results)} samples")

        # 保存汇总结果
        self._save_summary(results)

    def _load_json(self, filepath: str) -> List[Dict[str, Any]]:
        """加载 JSON 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        else:
            return [data]

    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """加载 JSONL 文件"""
        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _save_result(self, result: Dict[str, Any], idx: int) -> None:
        """保存单个结果"""
        output_path = os.path.join(
            self.dump_dir,
            f"prediction_{idx}.{self.output_format}"
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def _save_error(self, sample: Dict[str, Any], idx: int, error: str) -> None:
        """保存错误信息"""
        error_path = os.path.join(
            self.error_dir,
            f"error_{idx}.json"
        )

        error_data = {
            "sample": sample,
            "error": error,
        }

        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)

    def _save_summary(self, results: List[Dict[str, Any]]) -> None:
        """保存汇总结果"""
        summary_path = os.path.join(self.dump_dir, "summary.json")

        summary = {
            "total_samples": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results,
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to: {summary_path}")

    def run_interactive(self) -> None:
        """交互式推理"""
        logger.info("Starting interactive mode. Type 'quit' to exit.")

        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                messages = [{"role": "user", "content": user_input}]
                result = self.predict(messages)

                # 提取响应
                if "choices" in result and result["choices"]:
                    response = result["choices"][0]["message"]["content"]
                    print(f"\nAssistant: {response}")
                else:
                    print(f"\nError: {result}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        logger.info("Exiting interactive mode")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MedGemma Medical Inference Runner")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--input", type=str, help="Input file path (JSON/JSONL)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--model_path", type=str, help="Override model path")
    parser.add_argument("--dump_dir", type=str, help="Override output directory")

    args = parser.parse_args()

    # 加载配置
    configs = load_config(args.config)

    # 覆盖配置
    if args.model_path:
        configs.model.model_path = args.model_path
    if args.dump_dir:
        configs.output.dump_dir = args.dump_dir

    # 创建运行器
    runner = MedicalInferenceRunner(configs)

    # 运行推理
    if args.interactive:
        runner.run_interactive()
    elif args.input:
        runner.run_from_file(args.input)
    else:
        logger.error("Either --input or --interactive must be specified")


if __name__ == "__main__":
    main()
