import logging
import os
import shutil
from e3nn import o3
import torch.distributed as dist
import time

def check_args(args):
    """
    Check input arguments, update them if necessary for valid and consistent inputs, and return a tuple containing
    the (potentially) modified args and a list of log messages.
    """
    log_messages = []
    global_rank = int(os.environ.get("RANK",0))
    
    # Directories
    # Use work_dir for all other directories as well, unless they were specified by the user
    exp_path = os.path.join(args.output_dir,args.name)
    if global_rank == 0:
        if os.path.exists(exp_path):
            print(f"!!!Removing existing experiment directory: {exp_path}")
            shutil.rmtree(exp_path)  # 递归删除文件夹及内部所有文件
        os.makedirs(exp_path, exist_ok=True)        # 重新建立空文件夹
    else:
        time.sleep(3)

    dir_map = {
        "log_dir": "logs",
        "model_dir": "final_model",
        "checkpoints_dir": "checkpoints",
        "plot_dir": "plots",
        "downloads_dir": "downloads"}

    for attr, folder_name in dir_map.items():
        if getattr(args, attr) is None:
            setattr(args, attr, os.path.join(exp_path, folder_name))
            if global_rank == 0:
                os.makedirs(getattr(args,attr), exist_ok=True)

    # Model
    # Check if hidden_irreps, num_channels and max_L are consistent
    if args.hidden_irreps is None and args.num_channels is None and args.max_L is None:
        args.hidden_irreps, args.num_channels, args.max_L = "128x0e + 128x1o", 128, 1
    elif (
        args.hidden_irreps is not None
        and args.num_channels is not None
        and args.max_L is not None
    ):
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )
        log_messages.append(
            (
                "All of hidden_irreps, num_channels and max_L are specified",
                logging.WARNING,
            )
        )
        log_messages.append(
            (
                f"Using num_channels and max_L to create hidden_irreps: {args.hidden_irreps}.",
                logging.WARNING,
            )
        )
        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"
    elif args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )
        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"
    elif args.hidden_irreps is not None:
        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        args.num_channels = list(
            {irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}
        )[0]
        args.max_L = o3.Irreps(args.hidden_irreps).lmax
    elif args.max_L is not None and args.num_channels is None:
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.num_channels = 128
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )
    elif args.max_L is None and args.num_channels is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        args.max_L = 1
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    # Loss and optimization
    # Check Stage Two loss start
    if args.start_swa is not None:
        args.swa = True
        log_messages.append(
            (
                "Stage Two is activated as start_stage_two was defined",
                logging.INFO,
            )
        )

    if args.swa:
        if args.start_swa is None:
            args.start_swa = max(1, args.max_num_epochs // 4 * 3)
        if args.start_swa > args.max_num_epochs:
            log_messages.append(
                (
                    f"start_stage_two must be less than max_num_epochs, got {args.start_swa} > {args.max_num_epochs}",
                    logging.WARNING,
                )
            )
            log_messages.append(
                (
                    "Stage Two will not start, as start_stage_two > max_num_epochs",
                    logging.WARNING,
                )
            )
            args.swa = False

    return args, log_messages
