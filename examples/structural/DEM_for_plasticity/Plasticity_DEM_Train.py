import argparse

from DEM_Lib import *

from onescience.models.mlp import FullyConnectedNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--example", type=int, default=1, help="Example number to run (default: 1)"
)
args = parser.parse_args()
EXAMPLE = args.example

# Material properties
YM = 1000
PR = 0.3
sig_y0 = 50.0


def FlowStressLinear(eps_p_eff):
    return sig_y0 + (YM / 2.0) * eps_p_eff


def FlowStressKinematic(eps_p_eff):
    return sig_y0 + 0 * eps_p_eff


def HardeningModulusLinear(eps_p_eff):
    return YM / 2.0


# Setup examples
UNIFORM = True
if EXAMPLE == 1:
    print("Constrained shear")  # 约束剪切
    KINEMATIC = False
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear

    if UNIFORM:  # Get stress-strain curve
        ref_file = "shear10_iso"
        # ref_file = 'shear10_kine'
        KINEMATIC = True
        FlowStress = FlowStressKinematic

        disp_schedule = [
            0.0,
            1.0 / 3.0,
            2.0 / 3.0,
            1.0,
            2.0 / 3.0,
            1.0 / 3.0,
            0.0,
            -1.0 / 3.0,
            -2.0 / 3.0,
            -1.0,
            -2.0 / 3.0,
            -1.0 / 3.0,
            0.0,
        ]
        rel_tol = np.ones(13) * 1e-6

    else:
        ref_file = "shearWave_iso"
        # ref_file = 'ShearWave_kine'
        KINEMATIC = True
        FlowStress = FlowStressKinematic

        disp_schedule = [0.0, 0.5]
        rel_tol = [1e-6]

    BoundingBox = [4.0, 4.0, 1.0]  # Size of bounding box
    GeometryFile = "Shear"

elif EXAMPLE == 2:
    print("Bimaterial plate")  # 双材料板
    KINEMATIC = False
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear
    ref_file = "BiMat"
    disp_schedule = [0.0, 0.5]
    rel_tol = np.ones(1) * 1.0e-6
    BoundingBox = [4.0, 4.0, 1.0]  # Size of bounding box
    GeometryFile = "BiMat"

elif EXAMPLE == 3:
    print("Plate with hole, cyclic loading")  # 带孔板，循环加载
    ISO = True
    if ISO:
        KINEMATIC = False
        FlowStress = FlowStressLinear
        ref_file = "Hole"
        rel_tol = np.ones(4) * 2e-5
    else:
        KINEMATIC = True
        FlowStress = FlowStressKinematic
        ref_file = "HoleKine"
        rel_tol = np.ones(4) * 2e-5

    HardeningModulus = HardeningModulusLinear
    BoundingBox = [4.0, 4.0, 1.0]  # Size of bounding box
    GeometryFile = "Hole"
    disp_schedule = [0.0, 0.2, -0.2, 0.4, -0.4]

base = "./Example" + str(EXAMPLE) + "/"

# Setup domain
domain = setup_domain(base + GeometryFile, BoundingBox)
print("Number of nodes is ", domain["nN"])
print("Number of elements is ", domain["nE"])


# All misc model settings
step_max = len(disp_schedule) - 1
LBFGS_Iteration = 200
Num_Newton_itr = 1
Settings = [
    KINEMATIC,
    FlowStress,
    HardeningModulus,
    disp_schedule,
    rel_tol,
    step_max,
    LBFGS_Iteration,
    Num_Newton_itr,
    EXAMPLE,
    YM,
    PR,
    sig_y0,
    base,
    UNIFORM,
]

# Hyper parameters
x_var = {"x_lr": 0.05, "neuron": 100, "act_func": "tanh"}

lr = x_var["x_lr"]
H = int(x_var["neuron"])
act_fn = x_var["act_func"]
print("LR: " + str(lr) + ", H: " + str(H) + ", act fn: " + act_fn)


# Begin training

snet = FullyConnectedNet(
    in_features=3,
    layer_size=[128, 256, 512, 512, 256, 128],
    out_features=3,
    num_layers=6,
    activation_fn=act_fn,
)
DEM = DeepMixedMethod([snet, lr, domain, Settings])
all_diff = DEM.train_model(disp_schedule, ref_file)
fn_ = base + "AllDiff.npy"
np.save(fn_, all_diff)
