import argparse
import logging
import os

import torch
import train

from onescience.datapipes.transolver.dataset import GraphDataset
from onescience.datapipes.transolver.load_dataset import load_train_val_fold
from onescience.distributed.manager import DistributedManager
from onescience.models.transolver.Transolver3D import Transolver3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/mlcfd_data/training_data")
    parser.add_argument(
        "--preprocessed_save_dir", default="./dataset/mlcfd_data/preprocessed_data"
    )
    parser.add_argument("--model_save_dir", default="./metrics")
    parser.add_argument(
        "--fold_id",
        default=0,
        type=int,
        help="Which param folder should be selected as the test/validation set",
    )
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--val_iter", default=10, type=int)
    parser.add_argument("--cfd_config_dir", default="cfd/cfd_params.yaml")
    parser.add_argument("--cfd_model", default="Transolver")
    parser.add_argument("--cfd_mesh", action="store_true")
    parser.add_argument("--r", default=0.2, type=float)
    parser.add_argument(
        "--weight",
        default=0.5,
        type=float,
        help="Weight loss of pressure term. default=0.5",
    )
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--nb_epochs", default=200, type=int)
    parser.add_argument(
        "--preprocessed", default=1, type=int, help="Whether to use preprocessed data"
    )
    args = parser.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()

    # 只在rank 0进程配置日志，避免多卡打印重复信息
    if dist.rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        # 其他rank禁用日志输出
        logging.basicConfig(level=logging.ERROR)
    if dist.rank == 0:
        logging.info(f"Training arguments: {args}")

    if dist.world_size > 1:
        # 多卡分布式训练，使用 local_rank 对应的 GPU
        device = torch.device(
            f"cuda:{dist.local_rank}" if torch.cuda.is_available() else "cpu"
        )
    else:
        # 单卡训练，使用 --gpu 参数指定的 GPU
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
    logging.info(f"Using device: {device}")
    if dist.rank == 0:
        logging.info(f"World size: {dist.world_size}")

    torch.cuda.set_device(device)

    train_data, val_data, coef_norm = load_train_val_fold(
        args, preprocessed=args.preprocessed, dist=dist
    )
    train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
    val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

    if dist.rank == 0:
        logging.info(f"Number of training samples: {len(train_ds)}")

    if args.cfd_model == "Transolver":
        model = Transolver3D(
            n_hidden=256,
            n_layers=8,
            space_dim=7,
            fun_dim=0,
            n_head=8,
            mlp_ratio=2,
            out_dim=4,
            slice_num=32,
            unified_pos=0,
        )

    save_path = os.path.join(
        args.model_save_dir,
        args.cfd_model,
        str(args.fold_id),
        f"{args.nb_epochs}_{args.weight}",
    )
    if dist.rank == 0:
        os.makedirs(save_path, exist_ok=True)

    trained_model = train.main(
        device=device,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        hparams={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "nb_epochs": args.nb_epochs,
        },
        path=save_path,
        reg=args.weight,
        val_iter=args.val_iter,
        coef_norm=coef_norm,
    )
