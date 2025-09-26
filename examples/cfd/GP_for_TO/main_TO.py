import argparse
import time
from datetime import datetime

import torch
from train import find_TO

from onescience.models.GPs import GPPLUS
from onescience.utils.GP_TO import get_data_fluid, set_seed


def main(args):
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available(
            ) else "cpu"
        ),
    }

    # Fix Seed
    random_states = 11

    problem = args.problem  # 从参数获取problem

    ############################### Generate Data ##############################################
    set_seed(random_states)

    NOW = datetime.now()
    d = NOW.strftime("%B%d_%H-%M")

    N_train_per_BC = 25
    N_col_domain = 10000
    X_col, X_train, Sol_train = get_data_fluid(
        problem=problem, N_col_domain=N_col_domain, N_train=N_train_per_BC
    )
    mean_function = "neural_network"
    NN_archi = [64, 64, 64, 64, 64]

    # Names of the outputs
    output_names = ["u", "v", "p", "ro"]

    # Define all models as a list of models
    models = [
        GPPLUS(
            train_x=X_train[i].type(tkwargs["dtype"]),
            train_y=Sol_train[i].type(tkwargs["dtype"]),
            collocation_x=X_col.type(tkwargs["dtype"])
            .clone()
            .requires_grad_(True)
            .to(tkwargs["device"]),
            basis=mean_function,
            NN_layers_base=NN_archi,
            name_output=name,
            device=tkwargs["device"],
            dtype=tkwargs["dtype"],
        ).to(**tkwargs)
        for i, name in enumerate(output_names)
    ]

    ############################### Fit Model ##########################################
    start_time = time.time()  # get the current time
    loss_history = find_TO(
        model_list=models,
        num_iter=50000,
        lr_default=0.001,
        title=f"seed{random_states}_{problem}_{d}",
        problem=problem,
        diff_method="Numerical",
    )
    end_time = time.time()

    print(
        f"The total time in seconds is {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TO model training")
    parser.add_argument(
        "--problem",
        type=str,
        default="doublepipe",
        choices=["doublepipe", "diffuser",
                 "rugby", "pipebend"],
        help="Problem name to select the dataset and configuration",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU id to use, default=0")
    args = parser.parse_args()

    main(args)
