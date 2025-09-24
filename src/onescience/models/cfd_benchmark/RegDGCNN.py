import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import math
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    global_mean_pool,
    global_max_pool,
    JumpingKnowledge,
)
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import BatchNorm


def knn(x, k):
    """
    Computes the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
    """
    # Calculate pairwise distance, shape (batch_size, num_points, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # Retrieve the indices of the k nearest neighbors
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local graph features for each point by finding its k-nearest neighbors and
    concatenating the relative position vectors.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of neighbors to consider for graph construction.
        idx (torch.Tensor, optional): Precomputed k-nearest neighbor indices.

    Returns:
        torch.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    # Compute k-nearest neighbors if not provided
    if idx is None:
        idx = knn(x, k=k)

    # Prepare indices for gathering
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()

    # Gather neighbors for each point to construct local regions
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # Expand x to match the dimensions for broadcasting subtraction
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Concatenate the original point features with the relative positions to form the graph features
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


# class Model(nn.Module):
#     """
#     Deep Graph Convolutional Neural Network for Regression Tasks (RegDGCNN) for processing 3D point cloud data.

#     This network architecture extracts hierarchical features from point clouds using graph-based convolutions,
#     enabling effective learning of spatial structures.
#     """

#     def __init__(self, args, device):
#         """
#         Initializes the RegDGCNN model with specified configurations.

#         Args:
#             args (dict): Configuration parameters including 'k' for the number of neighbors, 'emb_dims' for embedding
#             dimensions, and 'dropout' rate.
#             output_channels (int): Number of output channels (e.g., for drag prediction, this is 1).
#         """
#         super(Model, self).__init__()
#         self.__name__ = 'RegDGCNN'
#         self.args = args
#         self.k = 20  # Number of nearest neighbors

#         # Batch normalization layers to stabilize and accelerate training
#         self.bn1 = nn.BatchNorm2d(args.n_hidden)
#         self.bn2 = nn.BatchNorm2d(args.n_hidden * 2)
#         self.bn3 = nn.BatchNorm2d(args.n_hidden * 2)
#         self.bn4 = nn.BatchNorm2d(args.n_hidden * 4)
#         self.bn5 = nn.BatchNorm1d(args.emb_dims)

#         # EdgeConv layers: Convolutional layers leveraging local neighborhood information
#         self.conv1 = nn.Sequential(nn.Conv2d(2*(args.fun_dim + args.space_dim), args.n_hidden, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(args.n_hidden * 2, args.n_hidden * 2, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(args.n_hidden * 4, args.n_hidden * 2, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(args.n_hidden * 4, args.n_hidden * 4, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(args.n_hidden * 9, args.emb_dims, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))

#         # Fully connected layers to interpret the extracted features and make predictions
#         self.linear1 = nn.Linear(args.emb_dims*2, 128, bias=False)
#         self.bn6 = nn.LayerNorm(128)
#         self.dp1 = nn.Dropout(p=args.dropout)

#         self.linear2 = nn.Linear(128, 64)
#         self.bn7 = nn.LayerNorm(64)
#         self.dp2 = nn.Dropout(p=args.dropout)

#         self.linear3 = nn.Linear(64, 32)
#         self.bn8 = nn.LayerNorm(32)
#         self.dp3 = nn.Dropout(p=args.dropout)

#         self.linear4 = nn.Linear(32, 16)
#         self.bn9 = nn.LayerNorm(16)
#         self.dp4 = nn.Dropout(p=args.dropout)

#         self.linear5 = nn.Linear(16, args.out_dim)  # The final output layer

#     def _make_norm_layer(self, channels):
#         """创建归一化层,支持batch size=1"""
#         return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

#     def forward(self, x, fx, T=None, geo=None):
#         """
#         Forward pass of the model to process input data and predict outputs.

#         Args:
#             x (torch.Tensor): Input tensor representing a batch of point clouds.

#         Returns:
#             torch.Tensor: Model predictions for the input batch.
#         """

#         batch_size = x.size(0)

#         # 兼容 batch_size = 1 输入：去除 batch 维度
#         x = torch.cat([x, fx], dim=-1)
#         x = x.permute(0, 2, 1)

#         print(f"x:{x.shape}")
#         # Extract graph features and apply EdgeConv blocks
#         x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
#         print(f"x:{x.shape}")
#         x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 256, num_points, k)

#         # Global max pooling
#         x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#         # Repeat the process for subsequent EdgeConv blocks
#         x = get_graph_feature(x1, k=self.k)   # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
#         x = self.conv2(x)                     # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
#         x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

#         x = get_graph_feature(x2, k=self.k)   # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
#         x = self.conv3(x)                     # (batch_size, 512*2, num_points, k) -> (batch_size, 512, num_points, k)
#         x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

#         x = get_graph_feature(x3, k=self.k)   # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
#         x = self.conv4(x)                     # (batch_size, 512*2, num_points, k) -> (batch_size, 1024, num_points, k)
#         x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points, k) -> (batch_size, 1024, num_points)

#         # Concatenate features from all EdgeConv blocks
#         x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 256+512+512+1024, num_points)

#         # Apply the final convolutional block
#         x = self.conv5(x)  # (batch_size, 256+512+512+1024, num_points) -> (batch_size, emb_dims, num_points)
#         # Combine global max and average pooling features
#         # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         x = torch.cat((x1, x2), 1)   # (batch_size, emb_dims*2)
#         print(f"x:{x.shape}")
#         # Process features through fully connected layers with dropout and batch normalization
#         x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
#         x = self.dp1(x)
#         x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
#         x = self.dp2(x)
#         x = F.leaky_relu(self.bn8(self.linear3(x)), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
#         x = self.dp3(x)
#         x = F.leaky_relu(self.bn9(self.linear4(x)), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
#         x = self.dp4(x)

#         # Final linear layer to produce the output
#         x = self.linear5(x)                                              # (batch_size, 16) -> (batch_size, 1)
#         print(f"x:{x.shape}")
#         x = x.permute(0, 2, 1)
#         return x


class Model(nn.Module):
    """
    Deep Graph Convolutional Neural Network for Regression Tasks (RegDGCNN) for processing 3D point cloud data.

    This network architecture extracts hierarchical features from point clouds using graph-based convolutions,
    enabling effective learning of spatial structures.
    """

    def __init__(self, args, device):
        """
        Initializes the RegDGCNN model with specified configurations.

        Args:
            args (dict): Configuration parameters including 'k' for the number of neighbors, 'emb_dims' for embedding
            dimensions, and 'dropout' rate.
            output_channels (int): Number of output channels (e.g., for drag prediction, this is 1).
        """
        super(Model, self).__init__()
        self.__name__ = "RegDGCNN"
        self.args = args
        self.k = 40  # Number of nearest neighbors

        # Batch normalization layers to stabilize and accelerate training
        self.bn1 = nn.BatchNorm2d(args.n_hidden)
        self.bn2 = nn.BatchNorm2d(args.n_hidden * 2)
        self.bn3 = nn.BatchNorm2d(args.n_hidden * 2)
        self.bn4 = nn.BatchNorm2d(args.n_hidden * 4)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        # EdgeConv layers: Convolutional layers leveraging local neighborhood information
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                2 * (args.fun_dim + args.space_dim),
                args.n_hidden,
                kernel_size=1,
                bias=False,
            ),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.n_hidden * 2, args.n_hidden * 2, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(args.n_hidden * 4, args.n_hidden * 2, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(args.n_hidden * 4, args.n_hidden * 4, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(args.n_hidden * 9, args.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Fully connected layers to interpret the extracted features and make predictions
        self.point_pred = nn.Sequential(
            nn.Linear(args.emb_dims, 64), nn.ReLU(), nn.Linear(64, args.out_dim)
        )

    def _make_norm_layer(self, channels):
        """创建归一化层,支持batch size=1"""
        return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

    def forward(self, x, fx, T=None, geo=None):
        """
        Forward pass of the model to process input data and predict outputs.

        Args:
            x (torch.Tensor): Input tensor representing a batch of point clouds.

        Returns:
            torch.Tensor: Model predictions for the input batch.
        """

        batch_size = x.size(0)

        # 兼容 batch_size = 1 输入：去除 batch 维度
        x = torch.cat([x, fx], dim=-1)
        x = x.permute(0, 2, 1)

        # Extract graph features and apply EdgeConv blocks
        x = get_graph_feature(
            x, k=self.k
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 256, num_points, k)
        # Global max pooling
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # Repeat the process for subsequent EdgeConv blocks
        x = get_graph_feature(
            x1, k=self.k
        )  # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = get_graph_feature(
            x2, k=self.k
        )  # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 512*2, num_points, k) -> (batch_size, 512, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = get_graph_feature(
            x3, k=self.k
        )  # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 512*2, num_points, k) -> (batch_size, 1024, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 1024, num_points, k) -> (batch_size, 1024, num_points)

        # Concatenate features from all EdgeConv blocks
        x = torch.cat(
            (x1, x2, x3, x4), dim=1
        )  # (batch_size, 256+512+512+1024, num_points)

        # Apply the final convolutional block
        x = self.conv5(
            x
        )  # (batch_size, 256+512+512+1024, num_points) -> (batch_size, emb_dims, num_points)
        x = x.permute(0, 2, 1)  # (batch_size, num_points, emb_dims)
        out = self.point_pred(x)  # 点级预测层，输出 (batch_size, num_points, out_dim)
        return out
