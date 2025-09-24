from .data_loader import (
    airfoil,
    ns,
    darcy,
    pipe,
    elas,
    plas,
    pdebench_autoregressive,
    pdebench_steady_darcy,
    car_design,
    cfd3d,
    BubbleTempVel,
    BubbleTemp,
)


def get_data(args, dist):
    data_dict = {
        "car_design": car_design,
        "pdebench_autoregressive": pdebench_autoregressive,
        "pdebench_steady_darcy": pdebench_steady_darcy,
        "elas": elas,
        "pipe": pipe,
        "airfoil": airfoil,
        "darcy": darcy,
        "ns": ns,
        "plas": plas,
        "cfd3d": cfd3d,
        "FB_Gravity_TempVel": BubbleTempVel,
        "FB_Gravity_Temp": BubbleTemp,
    }
    dataset = data_dict[args.loader](args, dist)
    train_loader, test_loader, shapelist = dataset.get_loader()
    return dataset, train_loader, test_loader, shapelist
