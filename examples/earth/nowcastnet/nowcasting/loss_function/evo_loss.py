import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolutionLoss(nn.Module):
    def __init__(self, configs):
        super(EvolutionLoss, self).__init__()
        self.lamda = configs.lamda
        self.configs = configs

    def weighted_distance(self, x_real, x_pred, weight):
        return torch.mean(torch.abs(x_real - x_pred) * weight)

    def weighted_l2_norm(self, grad, weight):
        return torch.mean((grad * torch.sqrt(weight)) ** 2)

    def motion_regularization(self, v_x, v_y, weight):
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).to(self.configs.device)
        sobel_y = sobel_x.T
        grad_v_x = torch.abs(
            F.conv2d(
                v_x.unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0), padding=1, groups=1
            )
        )
        grad_v_y = torch.abs(
            F.conv2d(
                v_y.unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0), padding=1, groups=1
            )
        )

        l2_norm_vx = self.weighted_l2_norm(grad_v_x, weight)
        l2_norm_vy = self.weighted_l2_norm(grad_v_y, weight)

        return l2_norm_vx + l2_norm_vy

    def forward(self, evo_result, evo_motion, real, v_x, v_y):
        J_accum = 0
        J_motion = 0

        length = self.configs.evo_ic
        for t in range(length):
            weight_t = torch.minimum(
                torch.full(real[:, t].shape, 24).to(
                    self.configs.device), 1 + real[:, t]
            )
            J_accum += self.weighted_distance(
                real[:, t], evo_result[:, t], weight_t)
            J_accum += self.weighted_distance(
                real[:, t], evo_motion[:, t], weight_t)

            J_motion += self.motion_regularization(
                v_x[:, t], v_y[:, t], weight_t)

        return J_accum + self.lamda * J_motion


if __name__ == "__main__":
    x_real = torch.rand(8, 10, 100, 100)
    x_pred = torch.rand(8, 10, 100, 100)
    x_moti = torch.rand(8, 10, 100, 100)
    v_x = torch.rand(8, 10, 100, 100)
    v_y = torch.rand(8, 10, 100, 100)

    evo = EvolutionLoss(configs)
    J_evolution = evo(x_real, x_moti, x_pred, v_x, v_y)
    print("J_evolution:", J_evolution)
