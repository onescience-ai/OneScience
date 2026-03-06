import logging
import os
import sys
import traceback
import urllib.request
from contextlib import nullcontext
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Mapping

import torch
import torch.distributed as dist

# 添加项目根目录到路径
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_path))

from onescience.models.protenix.config import parse_configs, parse_sys_args
from onescience.models.protenix.protenix import Protenix
from onescience.utils.protenix.distributed import DIST_WRAPPER
from onescience.utils.protenix.seed import seed_everything
from onescience.utils.protenix.torch_utils import to_device
from onescience.sciui.protenix.web_service.dependency_url import URL
from runner.dumper import DataDumper

# Biology 数据加载器导入
from onescience.datapipes.biology.dataloader import get_protein_dataloader
from onescience.utils.YParams import YParams

logger = logging.getLogger(__name__)


class BiologyInferenceRunner:
    """使用 Biology Dataloader 的推理运行器"""

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_dataloader()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        """初始化环境"""
        self.print(
            f"Distributed: world_size={DIST_WRAPPER.world_size}, "
            f"rank={DIST_WRAPPER.rank}, local_rank={DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")
        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        """初始化基础设置"""
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_dataloader(self) -> None:
        """初始化 Biology Dataloader"""
        logger.info("Initializing Biology Dataloader")
        self.dataloader = get_protein_dataloader(configs=self.configs)
        logger.info(f"Dataloader initialized with {len(self.dataloader.dataset)} samples")

    def init_model(self) -> None:
        """初始化模型"""
        self.model = Protenix(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        """加载检查点"""
        checkpoint_path = self.configs.load_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Checkpoint not found: {checkpoint_path}")
        self.print(f"Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, self.device)
        state_dict = checkpoint["model"]
        # Step 1: 去除 DDP 的 module. 前缀
        sample_key = [k for k in state_dict.keys()][0]
        if sample_key.startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        sample_key1 = [k for k in state_dict.keys()][101]
        if sample_key1.startswith("msa_module."):
            state_dict = {k.replace("msa_module.", "msa_module.msa."): v for k, v in state_dict.items()}
        # Step 2: 适配 input_embedder. → input_embedder.embedder. 的结构差异
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("input_embedder."):
                new_key = k.replace("input_embedder.", "input_embedder.embedder.")
            elif k.startswith("template_embedder.") and not k.startswith("template_embedder.embedder."):
                new_key = k.replace("template_embedder.", "template_embedder.embedder.")                
            elif k.startswith("relative_position_encoding.") and not k.startswith("relative_position_encoding.encoder."):
                new_key = k.replace("relative_position_encoding.", "relative_position_encoding.encoder.")
            elif new_key.startswith("msa_module.msa.blocks.") and ".pair_stack." in k and "Pairformer." not in k:
                new_key = k.replace(".pair_stack.", ".pair_stack.Pairformer.")   
            elif k.startswith("pairformer_stack.") and not k.startswith("pairformer_stack.Pairformer."):
                new_key = k.replace("pairformer_stack.", "pairformer_stack.Pairformer.")
            elif k.startswith("diffusion_module.") and not k.startswith("diffusion_module.Diffusion."):
                new_key = k.replace("diffusion_module.", "diffusion_module.Diffusion.")
            elif k.startswith("distogram_head.linear.") and not k.startswith("distogram_head.linear.Linear."):
                new_key = k.replace("distogram_head.linear.", "distogram_head.linear.Linear.")
            #elif k.startswith("confidence_head.pairformer_stack") and not k.startswith("confidence_head.pairformer_stack.Pairformer"):
            #     new_key = k.replace("confidence_head.pairformer_stack.", "confidence_head.pairformer_stack.Pairformer.")
            elif k.startswith("linear_no_bias_sinit.") and not k.startswith("linear_no_bias_sinit.Linear."):
                new_key = k.replace("linear_no_bias_sinit.", "linear_no_bias_sinit.Linear.")
            elif k.startswith("linear_no_bias_zinit1.") and not k.startswith("linear_no_bias_zinit1.Linear."):
                new_key = k.replace("linear_no_bias_zinit1.", "linear_no_bias_zinit1.Linear.")
            elif k.startswith("linear_no_bias_zinit2.") and not k.startswith("linear_no_bias_zinit2.Linear."):
                new_key = k.replace("linear_no_bias_zinit2.", "linear_no_bias_zinit2.Linear.")
            elif k.startswith("linear_no_bias_token_bond.") and not k.startswith("linear_no_bias_token_bond.Linear."):
                new_key = k.replace("linear_no_bias_token_bond.", "linear_no_bias_token_bond.Linear.")                
            elif k.startswith("linear_no_bias_z_cycle.") and not k.startswith("linear_no_bias_z_cycle.Linear."):
                new_key = k.replace("linear_no_bias_z_cycle.", "linear_no_bias_z_cycle.Linear.")                
            elif k.startswith("linear_no_bias_s.") and not k.startswith("linear_no_bias_s.Linear."):
                new_key = k.replace("linear_no_bias_s.", "linear_no_bias_s.Linear.") 
            else:
                # 如果不满足任何条件，则保留原始键名
                new_key = k    

            # 将修改后的键值对添加到新字典中
            new_state_dict[new_key] = v
        self.model.load_state_dict(new_state_dict, strict=self.configs.load_strict)
        self.model.eval()
        self.print("Checkpoint loaded")

    def init_dumper(self, need_atom_confidence=False, sorted_by_ranking_score=True):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Any]) -> dict:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available() else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )
        return prediction

    def print(self, message: str) -> None:
        if DIST_WRAPPER.rank == 0:
            logger.info(message)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def download_infercence_cache(configs: Any, model_version: str = "v0.5.0") -> None:
    for cache_name in ("ccd_components_file", "ccd_components_rdkit_mol_file"):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            logger.info(f"Downloading {cache_name} from {tos_url}")
            urllib.request.urlretrieve(tos_url, cur_cache_fpath)

    checkpoint_path = configs.load_checkpoint_path
    if not opexists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        tos_url = URL[f"model_{model_version}"]
        logger.info(f"Downloading model from {tos_url}")
        urllib.request.urlretrieve(tos_url, checkpoint_path)


def update_inference_configs(configs: Any, N_token: int):
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


def run_inference(runner: BiologyInferenceRunner, configs: Any) -> None:
    """运行推理"""
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)

        # 使用 dataloader 遍历数据
        for batch_idx, batch in enumerate(runner.dataloader):
            # batch 是列表，取第一个元素（因为 batch_size=1）
            data_item = batch[0] if isinstance(batch, list) else batch

            # 处理返回的元组格式 (result, atom_array, error_message)
            if isinstance(data_item, tuple):
                result_dict, atom_array, error_message = data_item
                sample_name = result_dict.get("sample_name", f"sample_{batch_idx}")
            else:
                # 兼容旧格式
                result_dict = data_item
                sample_name = data_item.get("sample_name", f"sample_{batch_idx}")
                atom_array = data_item.get("atom_array")
                error_message = ""

            try:
                if error_message:
                    raise RuntimeError(f"Dataset processing error: {error_message}")

                logger.info(f"[{batch_idx+1}/{len(runner.dataloader)}] Processing {sample_name}")

                # 从 dataloader 获取的数据中提取特征
                features_dict = result_dict["features"]
                if atom_array is None:
                    atom_array = result_dict.get("atom_array")

                # 构建输入数据
                N_token = features_dict["token_index"].shape[0]
                N_atom = features_dict["atom_to_token_idx"].shape[0]
                N_msa = features_dict["msa"].shape[0]
                N_asym = len(torch.unique(features_dict["asym_id"]))

                logger.info(
                    f"{sample_name}: N_asym={N_asym}, N_token={N_token}, "
                    f"N_atom={N_atom}, N_msa={N_msa}"
                )

                # 准备输入数据
                data = {
                    "input_feature_dict": features_dict,
                    "sample_name": sample_name,
                    "sample_index": batch_idx,
                }

                # 更新配置
                new_configs = update_inference_configs(configs, N_token)
                runner.update_model_configs(new_configs)

                # 预测
                prediction = runner.predict(data)

                # 保存结果
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=features_dict.get("entity_poly_type", {}),
                )

                logger.info(f"{sample_name} succeeded. Results saved to {configs.dump_dir}")
                torch.cuda.empty_cache()

            except Exception as e:
                error_message = f"{sample_name}: {e}\n{traceback.format_exc()}"
                logger.error(error_message)
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "w") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


def main():
    """主函数"""
    LOG_FORMAT = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 从YAML配置文件加载所有配置
    config_file_path = opjoin(os.path.dirname(current_path), "configs/inference_config.yaml")
    cfg = YParams(config_file_path, "inference", print_params=True)

    # 解析命令行参数（命令行参数优先级最高）
    configs = parse_configs(
        configs=cfg.params,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )

    # 下载必要文件
    download_infercence_cache(configs)

    # 运行推理
    logger.info("=" * 60)
    logger.info("Starting Biology Dataloader Inference")
    logger.info("=" * 60)

    runner = BiologyInferenceRunner(configs)
    run_inference(runner, configs)

    logger.info("=" * 60)
    logger.info("Inference Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
