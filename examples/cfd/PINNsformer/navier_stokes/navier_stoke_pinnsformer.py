import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from onescience.models.pinnsformer import PINNsformer2D
from onescience.utils.pinnsformer_util import *

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epochs = 1000
data = scipy.io.loadmat("./cylinder_nektar_wake.mat")

U_star = data["U_star"]  # N x 2 x T
P_star = data["p_star"]  # N x T
t_star = data["t"]  # T x 1
X_star = data["X_star"]  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T

UU = U_star[:, 0, :]  # N x T
VV = U_star[:, 1, :]  # N x T
PP = P_star  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

idx = np.random.choice(N * T, 2500, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]

x_train = np.expand_dims(np.tile(x_train[:], (5)), -1)
y_train = np.expand_dims(np.tile(y_train[:], (5)), -1)
t_train = make_time_sequence(t_train, num_step=5, step=1e-2)

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True).to(device)
v_train = torch.tensor(v_train, dtype=torch.float32, requires_grad=True).to(device)


class MyDataset(Dataset):
    def __init__(self, x, y, t, u, v):
        self.x = x
        self.y = y
        self.t = t
        self.u = u
        self.v = v

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx], self.u[idx], self.v[idx]


dataset = MyDataset(x_train, y_train, t_train, u_train, v_train)
batch_size = 1000
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model = PINNsformer2D(d_out=2, d_hidden=128, d_model=32, N=1, heads=2).to(device)
model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn="strong_wolfe")

n_params = get_n_params(model)

print(model)
print(get_n_params(model))

loss_track = []

for i in tqdm(range(1000)):

    def closure():
        psi_and_p = model(x_train, y_train, t_train)
        psi = psi_and_p[:, :, 0:1]
        p = psi_and_p[:, :, 1:2]

        u = torch.autograd.grad(
            psi,
            y_train,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True,
        )[0]
        v = -torch.autograd.grad(
            psi,
            x_train,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_t = torch.autograd.grad(
            u,
            t_train,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_x = torch.autograd.grad(
            u,
            x_train,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_y = torch.autograd.grad(
            u,
            y_train,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u,
            x_train,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            u,
            y_train,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        v_t = torch.autograd.grad(
            v,
            t_train,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_x = torch.autograd.grad(
            v,
            x_train,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_y = torch.autograd.grad(
            v,
            y_train,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_xx = torch.autograd.grad(
            v,
            x_train,
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_yy = torch.autograd.grad(
            v,
            y_train,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        p_x = torch.autograd.grad(
            p,
            x_train,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True,
        )[0]
        p_y = torch.autograd.grad(
            p,
            y_train,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True,
        )[0]

        f_u = u_t + (u * u_x + v * u_y) + p_x - 0.01 * (u_xx + u_yy)
        f_v = v_t + (u * v_x + v * v_y) + p_y - 0.01 * (v_xx + v_yy)

        loss = (
            torch.mean((u[:, 0] - u_train) ** 2)
            + torch.mean((v[:, 0] - v_train) ** 2)
            + torch.mean(f_u**2)
            + torch.mean(f_v**2)
        )

        loss_track.append(loss.item())

        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)
if not os.path.exists("./model"):
    os.makedirs("./model")
torch.save(model.state_dict(), "./model/ns_pinnsformer.pt")
# Test Data
snap = np.array([100])
x_star = X_star[:, 0:1]
y_star = X_star[:, 1:2]
t_star = TT[:, snap]

u_star = U_star[:, 0, snap]
v_star = U_star[:, 1, snap]
p_star = P_star[:, snap]

x_star = np.expand_dims(np.tile(x_star[:], (5)), -1)
y_star = np.expand_dims(np.tile(y_star[:], (5)), -1)
t_star = make_time_sequence(t_star, num_step=5, step=1e-2)

x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
y_star = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)

# with torch.no_grad():
psi_and_p = model(x_star, y_star, t_star)
psi = psi_and_p[:, :, 0:1]
p_pred = psi_and_p[:, :, 1:2]

u_pred = torch.autograd.grad(
    psi, x_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True
)[0]
v_pred = -torch.autograd.grad(
    psi, y_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True
)[0]

u_pred = u_pred.cpu().detach().numpy()[:, 0]
v_pred = v_pred.cpu().detach().numpy()[:, 0]
p_pred = p_pred.cpu().detach().numpy()[:, 0]


error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

if not os.path.exists("./result"):
    os.makedirs("./result")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# Predicted u(x,t)
im0 = axes[0].imshow((p_star).reshape(50, 100), extent=[-3, 8, -2, 2], aspect="auto")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Exact p(x,y)")
fig.colorbar(im0, ax=axes[0])

# Exact u(x,t)
im1 = axes[1].imshow((p_pred).reshape(50, 100), extent=[-3, 8, -2, 2], aspect="auto")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Predicted p(x,y)")
fig.colorbar(im1, ax=axes[1])

# Absolute Error
im2 = axes[2].imshow(
    np.abs(p_pred - p_star).reshape(50, 100), extent=[-3, 8, -2, 2], aspect="auto"
)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title("Absolute Error")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("./result/ns_pinnsformer.png")
