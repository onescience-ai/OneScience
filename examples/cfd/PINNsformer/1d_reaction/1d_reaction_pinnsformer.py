import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from tqdm import tqdm

from onescience.models.pinnsformer import PINNsformer1D
from onescience.utils.pinnsformer_util import *

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
epochs = 500

# Train PINNsformer
res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 101, 101)
res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

res = make_time_sequence(res, num_step=5, step=1e-4)
b_left = make_time_sequence(b_left, num_step=5, step=1e-4)
b_right = make_time_sequence(b_right, num_step=5, step=1e-4)
b_upper = make_time_sequence(b_upper, num_step=5, step=1e-4)
b_lower = make_time_sequence(b_lower, num_step=5, step=1e-4)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

x_res, t_res = res[:, :, 0:1], res[:, :, 1:2]
x_left, t_left = b_left[:, :, 0:1], b_left[:, :, 1:2]
x_right, t_right = b_right[:, :, 0:1], b_right[:, :, 1:2]
x_upper, t_upper = b_upper[:, :, 0:1], b_upper[:, :, 1:2]
x_lower, t_lower = b_lower[:, :, 0:1], b_lower[:, :, 1:2]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model = PINNsformer1D(d_out=1, d_hidden=128, d_model=32, N=1, heads=2).to(device)

model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn="strong_wolfe")

print(model)
print(get_n_params(model))

loss_track = []
pbar = tqdm(range(epochs))
for i in pbar:

    def closure():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(
            pred_res,
            x_res,
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_t = torch.autograd.grad(
            pred_res,
            t_res,
            grad_outputs=torch.ones_like(pred_res),
            retain_graph=True,
            create_graph=True,
        )[0]

        loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean(
            (
                pred_left[:, 0]
                - torch.exp(
                    -((x_left[:, 0] - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2)
                )
            )
            ** 2
        )

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

        loss = loss_res + loss_bc + loss_ic
        optim.zero_grad()
        loss.backward()
        return loss

    loss = optim.step(closure)
    pbar.set_postfix(loss=f"{loss.item():.6f}")

print(
    "Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}".format(
        loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]
    )
)
print("Train Loss: {:4f}".format(np.sum(loss_track[-1])))

if not os.path.exists("./model"):
    os.makedirs("./model")
torch.save(model.state_dict(), "./model/1dreaction_pinnsformer.pt")

# Visualize PINNsformer
res_test = make_time_sequence(res_test, num_step=5, step=1e-4)
res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
x_test, t_test = res_test[:, :, 0:1], res_test[:, :, 1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1]
    pred = pred.cpu().detach().numpy()

pred = pred.reshape(101, 101)


def h(x):
    return np.exp(-((x - np.pi) ** 2) / (2 * (np.pi / 4) ** 2))


def u_ana(x, t):
    return h(x) * np.exp(5 * t) / (h(x) * np.exp(5 * t) + 1 - h(x))


res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u**2))

print("relative L1 error: {:4f}".format(rl1))
print("relative L2 error: {:4f}".format(rl2))

if not os.path.exists("./result"):
    os.makedirs("./result")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1行3列，调整宽高比
# Predicted u(x,t)
im0 = axes[0].imshow(pred, extent=[0, np.pi * 2, 1, 0], aspect="auto")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
axes[0].set_title("Predicted u(x,t)")
fig.colorbar(im0, ax=axes[0])

# Exact u(x,t)
im1 = axes[1].imshow(u, extent=[0, np.pi * 2, 1, 0], aspect="auto")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].set_title("Exact u(x,t)")
fig.colorbar(im1, ax=axes[1])

# Absolute Error
im2 = axes[2].imshow(np.abs(pred - u), extent=[0, np.pi * 2, 1, 0], aspect="auto")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
axes[2].set_title("Absolute Error")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("./result/1dreaction_pinnsformer.png")
