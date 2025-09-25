"""
A Python script file for model execution, which adds:
- Training sections for different model modules,
- Parallel initialization,
- Necessary parameters,
compared to the official version that only supports inference.

"""

import argparse
import os

import nowcasting.evaluator as evaluator
import torch
import torch.distributed as dist
from model_factory import Model
from nowcasting.data_provider import datasets_factory

parser = argparse.ArgumentParser(description="NowcastNet")

parser.add_argument("--device", type=str, default="cpu:0")
parser.add_argument("--worker", type=int, default=1)
parser.add_argument("--cpu_worker", type=int, default=1)
parser.add_argument("--dataset_name", type=str, default="radar")
parser.add_argument("--input_length", type=int, default=9)
parser.add_argument("--total_length", type=int, default=29)
parser.add_argument("--img_height", type=int, default=512)
parser.add_argument("--img_width", type=int, default=512)
parser.add_argument("--img_ch", type=int, default=2)
parser.add_argument("--case_type", type=str, default="normal")
parser.add_argument("--model_name", type=str, default="nowcasting")
parser.add_argument("--gen_frm_dir", type=str, default="results/nowcasting")
parser.add_argument("--pretrained_model", type=str, default="")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_save_samples", type=int, default=10)
parser.add_argument("--ngf", type=int, default=32)
parser.add_argument("--dataset_path", type=str)

parser.add_argument("--do_train", type=bool, default=False)
parser.add_argument("--evo_pre", type=bool, default=False)
parser.add_argument("--gen_pre", type=bool, default=False)
parser.add_argument("--data_norm", type=bool, default=False)
parser.add_argument("--lamda", type=float, default=0.01)
parser.add_argument("--evo_lr", type=float, default=0.001)
parser.add_argument("--gen_lr", type=float, default=0.0003)
parser.add_argument("--dis_lr", type=float, default=0.00003)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--checkpoints_dir", type=str, default="./data/checkpoints/")
parser.add_argument("--slurm_id", type=str)

args = parser.parse_args()

args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.ic_feature = args.ngf * 10


def DDP_Init(batch_size):
    local_rank = 0
    world_size = 1
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    args.batch_size = int(batch_size / world_size)


def test_wrapper_pytorch_loader(model):
    batch_size_test = args.batch_size
    test_input_handle = datasets_factory.data_provider(args)
    args.batch_size = batch_size_test
    evaluator.test_pytorch_loader(model, test_input_handle, args, "test_result")


def train_wrapper_pytorch_loader(EvoNet, GenNet, DisNet):
    batch_size_test = args.batch_size
    train_input_handle, val_input_handle = datasets_factory.train_data_provider(args)
    args.batch_size = batch_size_test
    evaluator.train_pytorch_loader(
        EvoNet, GenNet, DisNet, train_input_handle, val_input_handle, args
    )


print("Initializing models")
print(
    f"do_train: {args.do_train}",
    f"gen_pre: {args.gen_pre}",
    f"gen_lr: {args.gen_lr}",
    f"dis_lr: {args.dis_lr}",
    "else: ",
)

if args.do_train:
    args.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    EvolutionNet = Model(args, mode="EvolutionNet")
    Generator = Model(args, mode="NowcastNet")
    Discriminator = Model(args, mode="Discriminator")
    train_wrapper_pytorch_loader(EvolutionNet, Generator, Discriminator)

else:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args, mode="Generator")
    test_wrapper_pytorch_loader(model)
