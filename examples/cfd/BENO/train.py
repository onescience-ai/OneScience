import argparse
import pickle
import pprint as pp
import random
import warnings
from timeit import default_timer

import matplotlib.tri as tri
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader, HeteroData
from torchvision.transforms import GaussianBlur

from onescience.distributed.manager import DistributedManager
from onescience.models.beno.BE_MPNN import HeteroGNS
from onescience.utils.beno.util import make_dir, record_data, to_cpu, to_np_array

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

DistributedManager.initialize()
dist = DistributedManager()

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--epochs", default=1000, type=int, help="Epochs")
parser.add_argument("--lr", default=0.00001, type=float, help="learning rate")
parser.add_argument(
    "--inspect_interval", default=100, type=int, help="inspect interval"
)
parser.add_argument("--id", default="0", type=str, help="ID")
parser.add_argument(
    "--init_boudary_loc",
    default="regular",
    type=str,
    help='choose from "random" or "regular" ',
)
parser.add_argument("--trans_layer", default=3, type=int, help="Layer of Transformer")
parser.add_argument(
    "--boundary_dim", default=128, type=int, help="Layer of Transformer"
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
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
pp.pprint(args.__dict__)


# dataset_type = args.dataset_type
# ## ===============================================================

DATA_PATH = f"./data/Dirichlet/"
f_all = np.load(DATA_PATH + "RHS_N32_4c_all.npy")
sol_all = np.load(DATA_PATH + "SOL_N32_4c_all.npy")
bc_all = np.load(DATA_PATH + "BC_N32_4c_all.npy")
ntrain = 900
ntest = 100
## ===============================================================


# DATA_PATH = f"./data/Dirichlet"
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

path = (
    "Resolution_"
    + str(s)
    + "_poisson"
    + "_ntrain"
    + str(ntrain)
    + "_kerwidth"
    + str(ker_width)
    + "_Transformer_layer"
    + str(args.trans_layer)
    + "_Rolling"
    + args.init_boudary_loc
    + "_ns"
    + str(args.ns)
    + "_nheads2"
    + "_bddim"
    + str(args.boundary_dim)
    + "_act"
    + args.act
    + "lr"
    + str(args.lr)
    + "_nmlp_layers"
    + str(args.nmlp_layers)
)
path_model = "./model/" + path

if dist.rank == 0:
    make_dir(path_model)
    print(f"Path: {path}")

cells_state = f_all[:, :, 3]  # node type \in {0,1,2,3}
coord_all = f_all[:, :, 0:2]  # all node corrdinate
bc_euco = bc_all[:, :, 0:2]  # boundary corrdinate
bc_value = bc_all[:, :, 2].reshape(-1, 128, 1)  # boundary value
bc_value = torch.tensor(bc_value)
bc_value_1 = bc_value[0:ntrain, :, :]
bc_euco = torch.tensor(bc_euco)
bcv_normalizer = GaussianNormalizer(bc_value_1)
bc_value = bcv_normalizer.encode(bc_value)
bc_euco = to_np_array(torch.cat([bc_euco, bc_value], dim=-1))
all_a = f_all[:, :, 2]
all_a_smooth = to_np_array(
    gblur(torch.tensor(all_a.reshape(all_a.shape[0], resolution, resolution))).flatten(
        start_dim=1
    )
)
all_a_reshape = all_a_smooth.reshape(-1, resolution, resolution)
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
        (all_a_reshape[:, :, 2:] - all_a_reshape[:, :, :-2]) / 2,
        all_a_reshape[:, :, -1:] - all_a_reshape[:, :, -2:-1],
    ],
    2,
)
all_a_grady = all_a_grady.reshape(-1, n)
all_u = sol_all[:, :, 0]

train_a = torch.FloatTensor(all_a[:ntrain])  # [num_train, 4096]
train_a_smooth = torch.FloatTensor(all_a_smooth[:ntrain])  # [num_train, 4096]
train_a_gradx = torch.FloatTensor(all_a_gradx[:ntrain])  # [num_train, 4096]
train_a_grady = torch.FloatTensor(all_a_grady[:ntrain])  # [num_train, 4096]
train_u = torch.FloatTensor(all_u[:ntrain])  # [num_train, 4096]
test_a = torch.FloatTensor(all_a[ntrain:])
test_a_smooth = torch.FloatTensor(all_a_smooth[ntrain:])
test_a_gradx = torch.FloatTensor(all_a_gradx[ntrain:])
test_a_grady = torch.FloatTensor(all_a_grady[ntrain:])
test_u = torch.FloatTensor(all_u[ntrain:])

bc_euco_train = bc_euco[:ntrain, :, :]
bc_euco_test = bc_euco[ntrain:, :, :]


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
        set([i for i in range(resolution * resolution)]) - set(list(outdomain_idx))
    )
    indomain_u = np.append(indomain_u, sol_all[j][indomain_idx])
    indomain_a = np.append(indomain_a, f_all[j][indomain_idx][:, 2])
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
meshgenerator = MeshGenerator([[0, 1], [0, 1]], [s, s], grid_input=grid_input)
data_train = []
for j in range(ntrain):
    mesh_idx_temp = [p for p in range(resolution**2)]
    outdomain_idx = np.array([])
    for p in range(f_all.shape[1]):
        if cells_state[j][p] != 0:
            outdomain_idx = np.append(outdomain_idx, p)
    for p in range(len(outdomain_idx)):
        mesh_idx_temp.remove(outdomain_idx[p])

    dist2bd_x = np.array([0, 0])[np.newaxis, :]
    dist2bd_y = np.array([0, 0])[np.newaxis, :]
    for p in range(len(mesh_idx_temp)):
        indomain_x = coord_all[j][mesh_idx_temp[p]][0]
        indomain_y = coord_all[j][mesh_idx_temp[p]][1]

        horizon_bd_y = np.where(bc_euco_train[j, :, 0].round(4) == indomain_x.round(4))[
            0
        ]
        dist2bd_y_temp = np.array(
            [
                np.abs(bc_euco_train[j, horizon_bd_y[0], 1] - indomain_y),
                np.abs(bc_euco_train[j, horizon_bd_y[1], 1] - indomain_y),
            ]
        )
        dist2bd_y = np.vstack([dist2bd_y, dist2bd_y_temp[np.newaxis, :]])
        horizon_bd_x = np.where(bc_euco_train[j, :, 1].round(4) == indomain_y.round(4))[
            0
        ]
        dist2bd_x_temp = np.array(
            [
                np.abs(bc_euco_train[j, horizon_bd_x[0], 0] - indomain_x),
                np.abs(bc_euco_train[j, horizon_bd_x[1], 0] - indomain_x),
            ]
        )
        dist2bd_x = np.vstack([dist2bd_x, dist2bd_x_temp[np.newaxis, :]])
    dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
    dist2bd_x = torch.tensor(dist2bd_x[1:]).float()

    idx = meshgenerator.sample(mesh_idx_temp)
    grid = meshgenerator.get_grid()

    xx = to_np_array(grid[:, 0])
    yy = to_np_array(grid[:, 1])
    triang = tri.Triangulation(xx, yy)
    tri_edge = triang.edges

    edge_index = meshgenerator.ball_connectivity(ns=10, tri_edge=tri_edge)
    edge_attr = meshgenerator.attributes(theta=train_a[j, :])
    train_x = torch.cat(
        [
            grid,
            train_a[j, idx].reshape(-1, 1),
            train_a_smooth[j, idx].reshape(-1, 1),
            train_a_gradx[j, idx].reshape(-1, 1),
            train_a_grady[j, idx].reshape(-1, 1),
            dist2bd_x,
            dist2bd_y,
        ],
        dim=1,
    )
    train_x_2 = torch.cat(
        [grid, torch.zeros([grid.shape[0], 4]), dist2bd_x, dist2bd_y], dim=1
    )

    bd_coord_input = torch.tensor(bc_euco_train[j])

    bd_coord_input_1 = bd_coord_input.clone()
    bd_coord_input_1[:, 2] = 0

    data = HeteroData()
    data["G1"].x = train_x  # node features ▲u=f
    data["G1"].boundary = bd_coord_input_1  # boundary value=0
    data["G1"].edge_features = edge_attr
    data["G1"].sample_idx = idx
    data["G1"].edge_index = edge_index

    data["G2"].x = train_x_2  ##node features ▲u=0
    data["G2"].boundary = bd_coord_input  # boundary value=g(x)
    data["G2"].edge_features = edge_attr
    data["G2"].sample_idx = idx
    data["G2"].edge_index = edge_index

    data["G1+2"].y = train_u[j, idx]

    data_train.append(data)

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
        indomain_x = coord_all[j + ntrain][mesh_idx_temp[p]][0]
        indomain_y = coord_all[j + ntrain][mesh_idx_temp[p]][1]

        horizon_bd_y = np.where(bc_euco_test[j, :, 0].round(4) == indomain_x.round(4))[
            0
        ]

        dist2bd_y_temp = np.array(
            [
                np.abs(bc_euco_test[j, horizon_bd_y[0], 1] - indomain_y),
                np.abs(bc_euco_test[j, horizon_bd_y[1], 1] - indomain_y),
            ]
        )
        dist2bd_y = np.vstack([dist2bd_y, dist2bd_y_temp[np.newaxis, :]])
        horizon_bd_x = np.where(bc_euco_test[j, :, 1].round(4) == indomain_y.round(4))[
            0
        ]

        dist2bd_x_temp = np.array(
            [
                np.abs(bc_euco_test[j, horizon_bd_x[0], 0] - indomain_x),
                np.abs(bc_euco_test[j, horizon_bd_x[1], 0] - indomain_x),
            ]
        )
        dist2bd_x = np.vstack([dist2bd_x, dist2bd_x_temp[np.newaxis, :]])
    dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
    dist2bd_x = torch.tensor(dist2bd_x[1:]).float()  # [num, 2]

    idx = meshgenerator.sample(mesh_idx_temp)
    grid = meshgenerator.get_grid()

    xx = to_np_array(grid[:, 0])
    yy = to_np_array(grid[:, 1])
    triang = tri.Triangulation(xx, yy)
    tri_edge = triang.edges

    edge_index = meshgenerator.ball_connectivity(ns=10, tri_edge=tri_edge)
    edge_attr = meshgenerator.attributes(theta=test_a[j, :])

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
    data["G1"].boundary = bd_coord_input_1  # boundary value=0
    data["G1"].edge_features = edge_attr
    data["G1"].sample_idx = idx
    data["G1"].edge_index = edge_index

    data["G2"].x = test_x_2  ##node features ▲u=0
    data["G2"].boundary = bd_coord_input  # boundary value=g(x)
    data["G2"].edge_features = edge_attr
    data["G2"].sample_idx = idx
    data["G2"].edge_index = edge_index

    data["G1+2"].y = test_u[j, idx]

    data_test.append(data)

train_sampler = (
    DistributedSampler(
        data_train, num_replicas=dist.world_size, rank=dist.rank, shuffle=True
    )
    if dist.world_size > 1
    else None
)
train_loader = DataLoader(
    data_train,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
)


test_loader = DataLoader(
    data_test, batch_size=batch_size2, shuffle=False, num_workers=0, pin_memory=True
)

t2 = default_timer()

if dist.rank == 0:
    print(f"preprocessing finished, time used:{t2 - t1}")
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
).to(dist.device)
if dist.world_size > 1:
    model = DistributedDataParallel(
        model,
        device_ids=[dist.local_rank],
        output_device=dist.device,
        broadcast_buffers=dist.broadcast_buffers,
        find_unused_parameters=dist.find_unused_parameters,
    )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=16, T_mult=2
)
myloss = LpLoss(size_average=False)
u_normalizer.cuda(dist.device)

if dist.world_size > 1:
    torch.distributed.barrier()
ttrain = np.zeros((epochs,))
ttest = np.zeros((epochs,))
model.train()

if dist.rank == 0:
    data_record = {}
else:
    data_record = None

for ep in range(epochs):

    if train_sampler is not None:
        train_sampler.set_epoch(ep)
    model.train()
    t1 = default_timer()

    train_mse = 0.0
    train_l2 = 0.0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(dist.device)
        optimizer.zero_grad()

        out = model(batch)
        loss = F.mse_loss(out.view(-1, 1), batch["G1+2"].y.view(-1, 1))

        loss.backward()

        l2 = myloss(
            u_normalizer.decode(
                out.view(batch_size, -1),
                sample_idx=batch["G1"].sample_idx.view(batch_size, -1),
            ),
            u_normalizer.decode(
                batch["G1+2"].y.view(batch_size, -1),
                sample_idx=batch["G1"].sample_idx.view(batch_size, -1),
            ),
        )

        optimizer.step()
        num_batches += 1
        train_mse += loss.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    # 同步指标
    train_mse_tensor = torch.tensor(train_mse, device=dist.device)
    train_l2_tensor = torch.tensor(train_l2, device=dist.device)
    num_batches_tensor = torch.tensor(num_batches, device=dist.device)

    # 全局求和
    if dist.world_size > 1:
        torch.distributed.all_reduce(
            train_mse_tensor, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(train_l2_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            num_batches_tensor, op=torch.distributed.ReduceOp.SUM
        )

    # 计算全局平均指标
    global_train_mse = train_mse_tensor.item() / num_batches_tensor.item()
    global_train_l2 = train_l2_tensor.item() / ntrain

    # 所有进程都参与测试
    model.eval()
    test_l2 = 0.0
    test_start = torch.cuda.Event(enable_timing=True)
    test_end = torch.cuda.Event(enable_timing=True)

    if dist.world_size > 1:
        torch.distributed.barrier()

    test_start.record()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dist.device)
            out = model(batch)

            # 只在rank 0计算指标
            if dist.rank == 0:
                out = u_normalizer.decode(
                    out.view(batch_size2, -1),
                    sample_idx=batch["G1"].sample_idx.view(batch_size2, -1),
                )
                test_l2 += myloss(out, batch["G1+2"].y.view(batch_size2, -1)).item()

    test_end.record()
    torch.cuda.synchronize()  # 等待测试完成

    # 同步测试结果
    test_l2_tensor = torch.tensor(test_l2, device=dist.device)
    if dist.world_size > 1:
        torch.distributed.all_reduce(test_l2_tensor, op=torch.distributed.ReduceOp.SUM)
    test_l2_normalized = test_l2_tensor.item() / ntest
    test_time_ms = test_start.elapsed_time(test_end) / 1000.0

    if dist.rank == 0:
        t3 = default_timer()
        total_time = t3 - t1
        ttrain[ep] = global_train_l2
        ttest[ep] = test_l2_normalized

        print(
            f"Epoch {ep:03d} | "
            f"Train MSE: {global_train_mse:.6f} | "
            f"Train L2: {global_train_l2:.6f} | "
            f"Test L2: {test_l2_normalized:.6f} | "
            f"Time: {total_time:.1f}s (Test: {test_time_ms:.1f}s)"
        )

        record_data(
            data_record,
            [ep, global_train_mse, global_train_l2, test_l2_normalized],
            ["epoch", "train_MSE", "train_L2", "test_L2"],
        )

        if ep % inspect_interval == 0 or ep == epochs - 1:
            model_state = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )
            record_data(
                data_record, [ep, to_cpu(model_state)], ["save_epoch", "state_dict"]
            )
            pickle.dump(data_record, open(path_model, "wb"))

# 添加同步点确保所有进程完成
if dist.world_size > 1:
    torch.distributed.barrier()

# 清理分布式进程
dist.cleanup()
