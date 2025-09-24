import torch
import torch.nn.functional as F
from einops import rearrange

import math


class LpLoss(object):
    def __init__(self, d=1, p=2, L=2 * math.pi, reduce_dims=0, reductions="sum"):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def rel(self, x, y):
        diff = torch.norm(
            torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, x, y):
        return self.rel(x, y)


class L2Loss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(L2Loss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def MSE(self, x, y):
        """
        Mean Squared Error
        """
        num_examples = x.size()[0]
        diff = (x.reshape(num_examples, -1) - y.reshape(num_examples, -1)) ** 2
        mse_per_sample = torch.mean(diff, dim=1)  # 每个样本的均方误差
        if self.reduction:
            if self.size_average:
                return torch.mean(mse_per_sample)
            else:
                return torch.sum(mse_per_sample)
        return mse_per_sample

    def MAE(self, x, y):
        """
        Mean Absolute Error
        """
        num_examples = x.size()[0]
        diff = torch.abs(x.view(num_examples, -1) - y.view(num_examples, -1))
        mae_per_sample = torch.mean(diff, dim=1)  # 每个样本的平均绝对误差
        if self.reduction:
            if self.size_average:
                return torch.mean(mae_per_sample)
            else:
                return torch.sum(mae_per_sample)
        return mae_per_sample

    def MaxAE(self, x, y):
        """
        Maximum Absolute Error
        """
        num_examples = x.size()[0]
        diff = torch.abs(x.view(num_examples, -1) - y.view(num_examples, -1))
        maxae_per_sample = torch.max(diff, dim=1)[0]  # 每个样本的最大绝对误差
        if self.reduction:
            if self.size_average:
                return torch.mean(maxae_per_sample)
            else:
                return torch.sum(maxae_per_sample)
        return maxae_per_sample

    def R2Score(self, x, y):
        """
        R^2 Score (Coefficient of Determination)
        """
        num_examples = x.size()[0]
        y_flat = y.view(num_examples, -1)
        x_flat = x.view(num_examples, -1)
        y_mean = torch.mean(
            y_flat, dim=1, keepdim=True
        )  # 每个样本真实值的均值，shape (num_examples, 1)

        ss_res = torch.sum((y_flat - x_flat) ** 2, dim=1)  # 残差平方和
        ss_tot = torch.sum((y_flat - y_mean) ** 2, dim=1)  # 总体平方和

        r2_per_sample = 1 - ss_res / (ss_tot + 1e-8)  # 为避免除零加小常数

        if self.reduction:
            if self.size_average:
                return torch.mean(r2_per_sample)
            else:
                return torch.sum(r2_per_sample)
        return r2_per_sample

    def __call__(self, x, y):
        return self.rel(x, y)


class DerivLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, shapelist=None):
        super(DerivLoss, self).__init__()

        assert d > 0 and p > 0
        self.shapelist = shapelist
        self.de_x = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)
        self.de_y = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)

    def central_diff(self, x, h1, h2, s1, s2):
        # assuming PBC
        # x: (batch, n, feats), h is the step size, assuming n = h*w
        x = rearrange(x, "b (h w) c -> b h w c", h=s1, w=s2)
        x = F.pad(x, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)  # [b c t h+2 w+2]
        grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (
            2 * h1
        )  # f(x+h) - f(x-h) / 2h
        grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (
            2 * h2
        )  # f(x+h) - f(x-h) / 2h

        return grad_x, grad_y

    def __call__(self, out, y):
        out = rearrange(
            out, "b (h w) c -> b c h w", h=self.shapelist[0], w=self.shapelist[1]
        )
        out = out[..., 1:-1, 1:-1].contiguous()
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out = rearrange(out, "b c h w -> b (h w) c")
        gt_grad_x, gt_grad_y = self.central_diff(
            y,
            1.0 / float(self.shapelist[0]),
            1.0 / float(self.shapelist[1]),
            self.shapelist[0],
            self.shapelist[1],
        )
        pred_grad_x, pred_grad_y = self.central_diff(
            out,
            1.0 / float(self.shapelist[0]),
            1.0 / float(self.shapelist[1]),
            self.shapelist[0],
            self.shapelist[1],
        )
        deriv_loss = self.de_x(pred_grad_x, gt_grad_x) + self.de_y(
            pred_grad_y, gt_grad_y
        )
        return deriv_loss
