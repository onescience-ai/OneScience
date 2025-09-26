import argparse
import pickle
import random
import warnings
from timeit import default_timer

import matplotlib.tri as tri
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, HeteroData
from torchvision.transforms import GaussianBlur

from onescience.models.beno.BE_MPNN import HeteroGNS
from onescience.utils.beno.util import record_data, to_np_array

# from utilities import *
from onescience.utils.beno.utilities import *

warnings.filterwarnings("ignore")
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="Testing")

parser.add_argument(
    "--dataset_type", default="32x32", type=str, help="dataset type")
parser.add_argument(
    "--epochs", default=1000, type=int, help="Epochs")
parser.add_argument(
    "--lr", default=0.00001, type=float, help="learning rate")
parser.add_argument(
    "--inspect_interval", default=100, type=int, help="inspect interval"
)
parser.add_argument("--id", default="0",
                    type=str, help="ID")
parser.add_argument(
    "--init_boudary_loc",
    default="regular",
    type=str,
    help='choose from "random" or "regular" ',
)
parser.add_argument("--trans_layer", default=3,
                    type=int, help="Layer of Transformer")
parser.add_argument(
    "--boundary_dim", default=128, type=int, help="Layer of Transformer"
)
parser.add_argument(
    "--batch_size", default=1, type=int, help="batch size")
parser.add_argument(
    "--act",
    default="relu",
    type=str,
    help='activation choose from "relu","elu","leakyrelu","silu',
)
parser.add_argument(
    "--nmlp_layers", default=2, type=int, help="number of layers of GNS"
)
parser.add_argument(
    "--ns", default=10, type=int, help="number of the number of neighbor nodes"
)


args = parser.parse_args()

# ## ===============================================================

DATA_PATH = f"./data/Dirichlet/"
f_all = np.load(DATA_PATH + "RHS_N32_4c_all.npy")
sol_all = np.load(DATA_PATH + "SOL_N32_4c_all.npy")
bc_all = np.load(DATA_PATH + "BC_N32_4c_all.npy")
ntrain = 900
ntest = 100
# ===============================================================

# ## ===============================================================

# DATA_PATH = f"./data/Neumann/"
# f_all = np.load(DATA_PATH + "RHS_N32_mix_all.npy")
# sol_all = np.load(DATA_PATH + "SOL_N32_mix_all.npy")
# bc_all=np.load(DATA_PATH + "BC_N32_mix_all.npy")
# ntrain = 900
# ntest =100

# ===============================================================

# DATA_PATH = f"./data/"
# f_all = np.load(DATA_PATH + "RHS_N32_10.npy")
# sol_all = np.load(DATA_PATH + "SOL_N32_10.npy")
# bc_all=np.load(DATA_PATH + "BC_N32_10.npy")
# ntrain = 7
# ntest =3

# ===============================================================


gblur = GaussianBlur(kernel_size=5, sigma=5)
batch_size = args.batch_size
batch_size2 = args.batch_size
width = 64
ker_width = 256
depth = 4
edge_features = 7
node_features = 10
ns = args.ns
epochs = args.epochs

learning_rate = args.lr
inspect_interval = args.inspect_interval

runtime = np.zeros(
    2,
)
t1 = default_timer()

resolution = 32
s = resolution
n = s**2


trans_layer = args.trans_layer


cells_state = f_all[:, :, 3]
coord_all = f_all[:, :, 0:2]
bc_euco = bc_all[:, :, 0:2]
bc_value = bc_all[:, :, 2].reshape(-1, 128, 1)
bc_value = torch.tensor(bc_value)
bc_value_1 = bc_value[0:ntrain, :, :]
bc_euco = torch.tensor(bc_euco)
bcv_normalizer = GaussianNormalizer(bc_value_1)
bc_value = bcv_normalizer.encode(bc_value)
bc_euco = to_np_array(
    torch.cat([bc_euco, bc_value], dim=-1))

all_a = f_all[:, :, 2]
all_a_smooth = to_np_array(
    gblur(torch.tensor(all_a.reshape(all_a.shape[0], resolution, resolution))).flatten(
        start_dim=1
    )
)
all_a_reshape = all_a_smooth.reshape(
    -1, resolution, resolution)
all_a_gradx = np.concatenate(
    [
        all_a_reshape[:, 1:2] - all_a_reshape[:, 0:1],
        (all_a_reshape[:, 2:] - all_a_reshape[:, :-2]) / 2,
        all_a_reshape[:, -1:] - all_a_reshape[:, -2:-1],
    ],
    1,
)
all_a_gradx = all_a_gradx.reshape(-1, n)
all_a_grady = np.concatenate(
    [
        all_a_reshape[:, :, 1:2] - all_a_reshape[:, :, 0:1],
        (all_a_reshape[:, :, 2:] -
         all_a_reshape[:, :, :-2]) / 2,
        all_a_reshape[:, :, -1:] -
        all_a_reshape[:, :, -2:-1],
    ],
    2,
)
all_a_grady = all_a_grady.reshape(-1, n)
all_u = sol_all[:, :, 0]

train_a = torch.FloatTensor(all_a[:ntrain])
train_a_smooth = torch.FloatTensor(all_a_smooth[:ntrain])
train_a_gradx = torch.FloatTensor(all_a_gradx[:ntrain])
train_a_grady = torch.FloatTensor(all_a_grady[:ntrain])
train_u = torch.FloatTensor(all_u[:ntrain])
test_a = torch.FloatTensor(all_a[ntrain:])
test_a_smooth = torch.FloatTensor(all_a_smooth[ntrain:])
test_a_gradx = torch.FloatTensor(all_a_gradx[ntrain:])
test_a_grady = torch.FloatTensor(all_a_grady[ntrain:])
test_u = torch.FloatTensor(all_u[ntrain:])
bc_euco_test = bc_euco[ntrain:, :, :]
cells_state_test = cells_state[ntrain:, :]

# * normalization
indomain_a = np.array([])
indomain_u = np.array([])
for j in range(ntrain):
    outdomain_idx = np.array([], dtype=int)
    indomain_idx = np.array([], dtype=int)
    for p in range(f_all.shape[1]):
        if cells_state[j][p] != 0:
            outdomain_idx = np.append(outdomain_idx, int(p))
    indomain_idx = list(
        set([i for i in range(resolution * resolution)]
            ) - set(list(outdomain_idx))
    )
    indomain_u = np.append(
        indomain_u, sol_all[j][indomain_idx])
    indomain_a = np.append(
        indomain_a, f_all[j][indomain_idx][:, 2])

indomain_u = torch.tensor(indomain_u)
indomain_a = torch.tensor(indomain_a)


a_normalizer = GaussianNormalizer(indomain_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)

u_normalizer = GaussianNormalizer(x=indomain_u)
train_u = u_normalizer.encode(train_u)

grid_input = f_all[-1, :, 0:2]
meshgenerator = MeshGenerator(
    [[0, 1], [0, 1]], [s, s], grid_input=grid_input)

data_test = []
for j in range(ntest):
    mesh_idx_temp = [p for p in range(resolution**2)]
    outdomain_idx = np.array([])
    for p in range(f_all.shape[1]):
        if cells_state[j + ntrain][p] != 0:
            outdomain_idx = np.append(outdomain_idx, p)

    for p in range(len(outdomain_idx)):
        mesh_idx_temp.remove(outdomain_idx[p])

    dist2bd_x = np.array([0, 0])[np.newaxis, :]
    dist2bd_y = np.array([0, 0])[np.newaxis, :]
    for p in range(len(mesh_idx_temp)):
        indomain_x = coord_all[j +
                               ntrain][mesh_idx_temp[p]][0]
        indomain_y = coord_all[j +
                               ntrain][mesh_idx_temp[p]][1]

        horizon_bd_y = np.where(bc_euco_test[j, :, 0].round(4) == indomain_x.round(4))[
            0
        ]

        dist2bd_y_temp = np.array(
            [
                np.abs(
                    bc_euco_test[j, horizon_bd_y[0], 1] - indomain_y),
                np.abs(
                    bc_euco_test[j, horizon_bd_y[1], 1] - indomain_y),
            ]
        )
        dist2bd_y = np.vstack(
            [dist2bd_y, dist2bd_y_temp[np.newaxis, :]])
        horizon_bd_x = np.where(bc_euco_test[j, :, 1].round(4) == indomain_y.round(4))[
            0
        ]

        dist2bd_x_temp = np.array(
            [
                np.abs(
                    bc_euco_test[j, horizon_bd_x[0], 0] - indomain_x),
                np.abs(
                    bc_euco_test[j, horizon_bd_x[1], 0] - indomain_x),
            ]
        )
        dist2bd_x = np.vstack(
            [dist2bd_x, dist2bd_x_temp[np.newaxis, :]])
    dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
    dist2bd_x = torch.tensor(dist2bd_x[1:]).float()

    idx = meshgenerator.sample(mesh_idx_temp)
    grid = meshgenerator.get_grid()
    xx = to_np_array(grid[:, 0])
    yy = to_np_array(grid[:, 1])
    triang = tri.Triangulation(xx, yy)
    tri_edge = triang.edges
    edge_index = meshgenerator.ball_connectivity(
        ns=10, tri_edge=tri_edge)
    edge_attr = meshgenerator.attributes(theta=test_a[j, :])
    cell_state_test_current = torch.FloatTensor(
        cells_state_test[j])
    test_x = torch.cat(
        [
            grid,
            test_a[j, idx].reshape(-1, 1),
            test_a_smooth[j, idx].reshape(-1, 1),
            test_a_gradx[j, idx].reshape(-1, 1),
            test_a_grady[j, idx].reshape(-1, 1),
            dist2bd_x,
            dist2bd_y,
        ],
        dim=1,
    )
    test_x_2 = torch.cat(
        [grid, torch.zeros([grid.shape[0], 4]), dist2bd_x, dist2bd_y], dim=1
    )
    bd_coord_input = torch.tensor(bc_euco_test[j])

    bd_coord_input_1 = bd_coord_input.clone()
    bd_coord_input_1[:, 2] = 0

    data = HeteroData()
    data["G1"].x = test_x  # node features ▲u=f
    # boundary value=0
    data["G1"].boundary = bd_coord_input_1
    data["G1"].edge_features = edge_attr
    data["G1"].sample_idx = idx
    data["G1"].edge_index = edge_index
    data["G1"].cell_state = cell_state_test_current

    data["G2"].x = test_x_2  # node features ▲u=0
    # boundary value=g(x)
    data["G2"].boundary = bd_coord_input
    data["G2"].edge_features = edge_attr
    data["G2"].sample_idx = idx
    data["G2"].edge_index = edge_index

    data["G1+2"].y = test_u[j, idx]

    data_test.append(data)

test_loader = DataLoader(
    data_test, batch_size=batch_size, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if args.act == "leakyrelu":
    activation = nn.LeakyReLU
elif args.act == "elu":
    activation = nn.ELU
elif args.act == "relu":
    activation = nn.ReLU
else:
    activation = nn.SiLU

model = HeteroGNS(
    nnode_in_features=node_features,
    nnode_out_features=1,
    nedge_in_features=edge_features,
    nmlp_layers=args.nmlp_layers,
    activation=activation,
    boundary_dim=args.boundary_dim,
    trans_layer=trans_layer,
).to(device)

# 检查模型参数
print("Node Features:", node_features)
print("Edge Features:", edge_features)
print("Boundary Dim:", args.boundary_dim)
print("Trans Layer:", trans_layer)

# 初始化模型
try:
    model = HeteroGNS(
        nnode_in_features=node_features,
        nnode_out_features=1,
        nedge_in_features=edge_features,
        nmlp_layers=args.nmlp_layers,
        activation=activation,
        boundary_dim=args.boundary_dim,
        trans_layer=trans_layer,
    ).to(device)
    print("Model initialized successfully.")
except Exception as e:
    print("Error during model initialization:", str(e))


filename_model = "1Resolution_32_poisson_ntrain900_kerwidth256_Transformer_layer3_Rollingregular_ns10_nheads2_bddim128_actrelulr1e-05_nmlp_layers2"


myloss = LpLoss(size_average=False)
u_normalizer.cuda(device)

data_record = pickle.load(
    open(f"./model/{filename_model}", "rb"))

model.load_state_dict(data_record["state_dict"][-1])
analysis_record = {}

model.eval()
out_all = np.array([])
label_all = np.array([])
a_ori_all = np.array([])
mask_all = np.array([])
grid_all = np.array([])
with torch.no_grad():
    for ii, data in enumerate(test_loader):
        data = data.to(device)
        out_indomain = model(data)

        data_all = torch.zeros(
            (resolution * resolution, 10)).to(device)
        out = torch.zeros(
            (resolution * resolution, 1)).to(device)
        label = torch.zeros(
            (resolution * resolution)).to(device)
        grid_info = torch.zeros(
            (resolution * resolution, 2)).to(device)

        out[data["G1"].sample_idx] = out_indomain
        label[data["G1"].sample_idx] = data["G1+2"].y
        data_all[data["G1"].sample_idx, :] = data["G1"].x

        grid_info[data["G1"].sample_idx,
                  :] = data["G1"].x[:, :2]

        out = u_normalizer.decode(out.view(batch_size2, -1))

        out_tem = torch.zeros(
            (1, resolution * resolution)).to(device)
        out_tem[0][data["G1"].sample_idx] = out[0][data["G1"].sample_idx]
        cell_state_tem = torch.zeros(
            (1, resolution * resolution)).to(device)
        cell_state_tem[0, :] = data["G1"].cell_state

        a_ori = a_normalizer.decode(
            data_all[:, 2].view(1, -1))
        a_ori_tem = torch.zeros(
            (1, resolution * resolution)).to(device)
        a_ori_tem[0][data["G1"].sample_idx] = a_ori[0][data["G1"].sample_idx]

        grid_tem = torch.zeros(
            (1, resolution * resolution, 2)).to(device)
        grid_tem[0][data["G1"].sample_idx,
                    :] = grid_info[data["G1"].sample_idx, :]

        l2_item = myloss(out_tem, label.view(
            batch_size2, -1)).item()
        mae_item = nn.L1Loss()(
            out_tem, label.view(batch_size2, -1)).item()
        record_data(analysis_record, [
                    l2_item, mae_item], ["L2", "MAE"])
        out_all = np.append(out_all, to_np_array(out_tem))
        label_all = np.append(label_all, to_np_array(label))
        a_ori_all = np.append(
            a_ori_all, to_np_array(a_ori_tem))
        mask_all = np.append(
            mask_all, to_np_array(cell_state_tem))
        grid_all = np.append(
            grid_all, to_np_array(grid_tem))

plot_data(
    predict_term=out_all,
    true_term=label_all,
    forcing_term=a_ori_all,
    forcing_mask=mask_all,
    grid_info=grid_all,
    resolution=resolution,
    num_samples=3,
    interpolation="bilinear",
    save_path="./picture/forcing_solution_comparison.png",
)

print(f"Mean L2 Loss: {np.mean(analysis_record['L2'])}")
print(f"Std L2 Loss: {np.std(analysis_record['L2'])}")
print(f"Mean MAE Loss: {np.mean(analysis_record['MAE'])}")
print(f"Std MAE Loss: {np.std(analysis_record['MAE'])}")
