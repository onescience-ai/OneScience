import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import scipy.io as scio
from .shapenet_utils import get_datalist
from .shapenet_utils import GraphDataset, GraphDataset_mgn
from torch.utils.data import Dataset, ConcatDataset
from onescience.utils.cfd_benchmark.normalizer import (
    UnitTransformer,
    UnitGaussianNormalizer,
)
import dgl
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import random
import torchvision.transforms.functional as TF


class plas(object):
    def __init__(self, args, dist):
        self.DATA_PATH = args.data_path + "/plas_N987_T20.mat"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_out = args.T_out
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.rank = dist.rank
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def random_collate_fn(self, batch):
        shuffled_batch = []
        shuffled_u = None
        shuffled_t = None
        shuffled_a = None
        shuffled_pos = None
        for item in batch:
            pos = item[0]
            t = item[1]
            a = item[2]
            u = item[3]

            num_timesteps = t.size(0)
            permuted_indices = torch.randperm(num_timesteps)
            t = t[permuted_indices]
            u = u.reshape(u.shape[0], num_timesteps, -1)[
                ..., permuted_indices, :
            ].reshape(u.shape[0], -1)

            if shuffled_t is None:
                shuffled_pos = pos.unsqueeze(0)
                shuffled_t = t.unsqueeze(0)
                shuffled_u = u.unsqueeze(0)
                shuffled_a = a.unsqueeze(0)
            else:
                shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
                shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
                shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
                shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

        shuffled_batch.append(shuffled_pos)
        shuffled_batch.append(shuffled_t)
        shuffled_batch.append(shuffled_a)
        shuffled_batch.append(shuffled_u)

        return shuffled_batch  # B N T 4

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((101 - 1) / r1) + 1)
        s2 = int(((31 - 1) / r2) + 1)

        data = scio.loadmat(self.DATA_PATH)
        input = torch.tensor(data["input"], dtype=torch.float)
        output = torch.tensor(data["output"], dtype=torch.float)
        x_train = (
            input[: self.ntrain, ::r1][:, :s1]
            .reshape(self.ntrain, s1, 1)
            .repeat(1, 1, s2)
        )
        x_train = x_train.reshape(self.ntrain, -1, 1)
        y_train = output[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, self.T_out * self.out_dim)
        x_test = (
            input[-self.ntest :, ::r1][:, :s1]
            .reshape(self.ntest, s1, 1)
            .repeat(1, 1, s2)
        )
        x_test = x_test.reshape(self.ntest, -1, 1)
        y_test = output[-self.ntest :, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, self.T_out * self.out_dim)

        # Use appropriate normalizer based on norm_type
        if self.norm_type == "UnitTransformer":
            self.x_normalizer = UnitTransformer(x_train)
        elif self.norm_type == "UnitGaussianNormalizer":
            self.x_normalizer = UnitGaussianNormalizer(x_train)

        x_train = self.x_normalizer.encode(x_train)
        x_test = self.x_normalizer.encode(x_test)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            y_train = self.y_normalizer.encode(y_train)

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        t = np.linspace(0, 1, self.T_out)
        t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
        t_train = t.repeat(self.ntrain, 1)
        t_test = t.repeat(self.ntest, 1)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.random_collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class elas(object):
    def __init__(self, args, dist):
        self.PATH_Sigma = args.data_path + "/Meshes/Random_UnitCell_sigma_10.npy"
        self.PATH_XY = args.data_path + "/Meshes/Random_UnitCell_XY_10.npy"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.rank = dist.rank
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        input_s = np.load(self.PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
        input_xy = np.load(self.PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
        print(f"input_s:{input_s.shape},input_xy:{input_xy.shape}")
        train_s = input_s[: self.ntrain, :, None]
        test_s = input_s[-self.ntest :, :, None]
        train_xy = input_xy[: self.ntrain]
        test_xy = input_xy[-self.ntest :]

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(train_s)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.y_normalizer = UnitGaussianNormalizer(train_s)

            train_s = self.y_normalizer.encode(train_s)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [train_s.shape[1]]


class pipe(object):
    def __init__(self, args, dist):
        self.INPUT_X = args.data_path + "/Pipe_X.npy"
        self.INPUT_Y = args.data_path + "/Pipe_Y.npy"
        self.OUTPUT_Sigma = args.data_path + "/Pipe_Q.npy"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.rank = dist.rank
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((129 - 1) / r1) + 1)
        s2 = int(((129 - 1) / r2) + 1)

        inputX = np.load(self.INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(self.INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(self.OUTPUT_Sigma)[:, 0]
        output = torch.tensor(output, dtype=torch.float)

        x_train = input[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class airfoil(object):
    def __init__(self, args, dist):
        self.INPUT_X = args.data_path + "/NACA_Cylinder_X.npy"
        self.INPUT_Y = args.data_path + "/NACA_Cylinder_Y.npy"
        self.OUTPUT_Sigma = args.data_path + "/NACA_Cylinder_Q.npy"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.rank = dist.rank
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((221 - 1) / r1) + 1)
        s2 = int(((51 - 1) / r2) + 1)

        inputX = np.load(self.INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(self.INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(self.OUTPUT_Sigma)[:, 4]
        output = torch.tensor(output, dtype=torch.float)
        x_train = input[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class darcy(object):
    def __init__(self, args, dist):
        self.train_path = args.data_path + "/piececonst_r421_N1024_smooth1.mat"
        self.test_path = args.data_path + "/piececonst_r421_N1024_smooth2.mat"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.rank = dist.rank
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((421 - 1) / r1) + 1)
        s2 = int(((421 - 1) / r2) + 1)

        train_data = scio.loadmat(self.train_path)
        x_train = train_data["coeff"][: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 1)
        x_train = torch.from_numpy(x_train).float()
        y_train = train_data["sol"][: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_train = torch.from_numpy(y_train)

        test_data = scio.loadmat(self.test_path)
        x_test = test_data["coeff"][: self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_test = x_test.reshape(self.ntest, -1, 1)
        x_test = torch.from_numpy(x_test).float()
        y_test = test_data["sol"][: self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, 1)
        y_test = torch.from_numpy(y_test)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class ns(object):
    def __init__(self, args, dist):
        self.data_path = args.data_path + "/NavierStokes_V1e-5_N1200_T20.mat"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.rank = dist.rank
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((64 - 1) / r1) + 1)
        s2 = int(((64 - 1) / r2) + 1)

        data = scio.loadmat(self.data_path)
        print(data["u"].shape, s1, s2)
        train_a = data["u"][: self.ntrain, ::r1, ::r2, None, : self.T_in][
            :, :s1, :s2, :, :
        ]
        train_a = train_a.reshape(
            train_a.shape[0], -1, self.out_dim * train_a.shape[-1]
        )
        train_a = torch.from_numpy(train_a)
        train_u = data["u"][
            : self.ntrain, ::r1, ::r2, None, self.T_in : self.T_out + self.T_in
        ][:, :s1, :s2, :, :]
        train_u = train_u.reshape(
            train_u.shape[0], -1, self.out_dim * train_u.shape[-1]
        )
        train_u = torch.from_numpy(train_u)

        test_a = data["u"][-self.ntest :, ::r1, ::r2, None, : self.T_in][
            :, :s1, :s2, :, :
        ]
        test_a = test_a.reshape(test_a.shape[0], -1, self.out_dim * test_a.shape[-1])
        test_a = torch.from_numpy(test_a)
        test_u = data["u"][
            -self.ntest :, ::r1, ::r2, None, self.T_in : self.T_out + self.T_in
        ][:, :s1, :s2, :, :]
        test_u = test_u.reshape(test_u.shape[0], -1, self.out_dim * test_u.shape[-1])
        test_u = torch.from_numpy(test_u)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(train_a)
                self.y_normalizer = UnitTransformer(train_u)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(train_a)
                self.y_normalizer = UnitGaussianNormalizer(train_u)

            train_a = self.x_normalizer.encode(train_a)
            test_a = self.x_normalizer.encode(test_a)
            train_u = self.y_normalizer.encode(train_u)

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, train_a, train_u),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, test_a, test_u),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class pdebench_autoregressive(object):
    def __init__(self, args, dist):
        self.file_path = args.data_path
        self.train_ratio = args.train_ratio
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.batch_size = args.batch_size
        self.out_dim = args.out_dim
        self.rank = dist.rank

    def get_loader(self):
        train_dataset = pdebench_dataset_autoregressive(
            file_path=self.file_path,
            train_ratio=self.train_ratio,
            test=False,
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
        )
        test_dataset = pdebench_dataset_autoregressive(
            file_path=self.file_path,
            train_ratio=self.train_ratio,
            test=True,
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, train_dataset.shapelist


class pdebench_dataset_autoregressive(Dataset):
    def __init__(
        self,
        file_path: str,
        train_ratio: int,
        test: bool,
        T_in: int,
        T_out: int,
        out_dim: int,
    ):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())
            self.shapelist = h5_file[data_list[0]]["data"].shape[
                1:-1
            ]  # obtain shapelist
        self.ntrain = int(len(data_list) * train_ratio)
        self.test = test
        if not self.test:
            self.data_list = data_list[: self.ntrain]
        else:
            self.data_list = data_list[self.ntrain :]
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim
        self.rank = dist.rank

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as h5_file:
            data_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(data_group["data"], dtype="f")
            dim = len(data.shape) - 2
            T, *_, V = data.shape
            # change data shape
            data = (
                torch.tensor(data, dtype=torch.float)
                .movedim(0, -2)
                .contiguous()
                .reshape(*self.shapelist, -1)
            )
            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(data_group["grid"]["x"], dtype="f")
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(data_group["grid"]["x"], dtype="f")
                y = np.array(data_group["grid"]["y"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing="ij")
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(data_group["grid"]["x"], dtype="f")
                y = np.array(data_group["grid"]["y"], dtype="f")
                z = np.array(data_group["grid"]["z"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                grid = torch.stack((X, Y, Z), axis=-1)

        return (
            grid,
            data[:, : self.T_in * self.out_dim],
            data[
                :, (self.T_in) * self.out_dim : (self.T_in + self.T_out) * self.out_dim
            ],
        )


class pdebench_steady_darcy(object):
    def __init__(self, args, dist):
        self.file_path = args.data_path
        self.ntrain = args.ntrain
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.rank = dist.rank

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((128 - 1) / r1) + 1)
        s2 = int(((128 - 1) / r2) + 1)
        with h5py.File(self.file_path, "r") as h5_file:
            data_nu = np.array(h5_file["nu"], dtype="f")[:, ::r1, ::r2][:, :s1, :s2]
            data_solution = np.array(h5_file["tensor"], dtype="f")[:, :, ::r1, ::r2][
                :, :, :s1, :s2
            ]
            data_nu = torch.from_numpy(data_nu)
            data_solution = torch.from_numpy(data_solution)
            x = np.array(h5_file["x-coordinate"])
            y = np.array(h5_file["y-coordinate"])
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            grid = torch.stack((X, Y), axis=-1)[None, ::r1, ::r2, :][:, :s1, :s2, :]

        grid = grid.repeat(data_nu.shape[0], 1, 1, 1)

        pos_train = grid[: self.ntrain, :, :, :].reshape(self.ntrain, -1, 2)
        x_train = data_nu[: self.ntrain, :, :].reshape(self.ntrain, -1, 1)
        y_train = data_solution[: self.ntrain, 0, :, :].reshape(
            self.ntrain, -1, 1
        )  # solutions only have 1 channel

        pos_test = grid[self.ntrain :, :, :, :].reshape(
            data_nu.shape[0] - self.ntrain, -1, 2
        )
        x_test = data_nu[self.ntrain :, :, :].reshape(
            data_nu.shape[0] - self.ntrain, -1, 1
        )
        y_test = data_solution[self.ntrain :, 0, :, :].reshape(
            data_nu.shape[0] - self.ntrain, -1, 1
        )  # solutions only have 1 channel

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class car_design(object):
    def __init__(self, args, dist):
        self.file_path = args.data_path
        self.radius = args.radius
        self.test_fold_id = 0
        self.batch_size = args.batch_size
        self.rank = dist.rank
        self.model = args.model

    def get_samples(self, obj_path):
        folds = [f"param{i}" for i in range(9)]
        samples = []
        for fold in folds:
            fold_samples = []
            files = os.listdir(os.path.join(obj_path, fold))
            for file in files:
                path = os.path.join(obj_path, os.path.join(fold, file))
                if os.path.isdir(path):
                    fold_samples.append(os.path.join(fold, file))
            samples.append(fold_samples)
        return samples  # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples

    def load_train_val_fold(self):
        samples = self.get_samples(os.path.join(self.file_path, "training_data"))
        trainlst = []
        for i in range(len(samples)):
            if i == self.test_fold_id:
                continue
            trainlst += samples[i]
        vallst = (
            samples[self.test_fold_id]
            if 0 <= self.test_fold_id < len(samples)
            else None
        )

        if os.path.exists(os.path.join(self.file_path, "preprocessed_data")):
            if self.rank == 0:
                print("use preprocessed data")
            preprocessed = True
        else:
            preprocessed = False
        if self.rank == 0:
            print("loading data")
        train_dataset, coef_norm = get_datalist(
            self.file_path,
            trainlst,
            norm=True,
            savedir=os.path.join(self.file_path, "preprocessed_data"),
            preprocessed=preprocessed,
        )
        test_dataset = get_datalist(
            self.file_path,
            vallst,
            coef_norm=coef_norm,
            savedir=os.path.join(self.file_path, "preprocessed_data"),
            preprocessed=preprocessed,
        )
        return train_dataset, test_dataset, coef_norm, vallst

    def collate_fn(self, batch):
        """处理DGL图批处理的函数"""
        # 将PyG批处理转换为DGL图列表
        graphs = []
        edge_features = []
        all_data = []
        obj_files = []

        for data in batch:
            # 创建DGL图
            g = dgl.graph(
                (data.edge_index[0], data.edge_index[1]), num_nodes=data.x.shape[0]
            )
            graphs.append(g)
            edge_features.append(data.edge_attr)
            all_data.append(
                {"x": data.x, "y": data.y, "surf": data.surf, "pos": data.pos}
            )
            obj_files.append(data.obj_file)
        batched_graph = dgl.batch(graphs)

        return {
            "graphs": batched_graph,
            "node_features": torch.cat([d["x"] for d in all_data]),
            "edge_features": torch.cat(edge_features),
            "labels": torch.cat([d["y"] for d in all_data]),
            "surf_mask": torch.cat([d["surf"] for d in all_data]),
            "pos": torch.cat([d["pos"] for d in all_data]),
            "obj_files": obj_files,  # 新增：返回对象文件列表
        }

    def get_loader(self):
        train_data, val_data, coef_norm, vallst = self.load_train_val_fold()
        self.coef_norm = coef_norm

        # 创建数据集

        if self.model == "MeshGraphNet":
            train_dataset = GraphDataset_mgn(
                train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm
            )
            test_dataset = GraphDataset_mgn(
                val_data,
                use_cfd_mesh=False,
                r=self.radius,
                coef_norm=coef_norm,
                valid_list=vallst,
            )
            # 创建带批处理的DataLoader
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
        else:
            train_dataset = GraphDataset(
                train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm
            )
            test_dataset = GraphDataset(
                val_data,
                use_cfd_mesh=False,
                r=self.radius,
                coef_norm=coef_norm,
                valid_list=vallst,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [train_data[0].x.shape[0]]


class cfd_3d_dataset(Dataset):
    def __init__(
        self,
        data_path,
        downsamplex,
        downsampley,
        downsamplez,
        T_in,
        T_out,
        out_dim,
        is_train=True,
        train_ratio=0.8,
    ):
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim
        self.is_train = is_train

        # Calculate grid sizes
        self.r1 = downsamplex
        self.r2 = downsampley
        self.r3 = downsamplez
        self.s1 = int(((128 - 1) / self.r1) + 1)
        self.s2 = int(((128 - 1) / self.r2) + 1)
        self.s3 = int(((128 - 1) / self.r3) + 1)
        # Create position grid once (reused for all samples)
        with h5py.File(data_path, "r") as h5_file:
            x_coords = np.array(h5_file["x-coordinate"][:: self.r1])[: self.s1]
            y_coords = np.array(h5_file["y-coordinate"][:: self.r2])[: self.s2]
            z_coords = np.array(h5_file["z-coordinate"][:: self.r3])[: self.s3]

            # Create grid
            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            z = torch.tensor(z_coords, dtype=torch.float)
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            self.grid = torch.stack((X, Y, Z), axis=-1)
            self.grid_flat = self.grid.reshape(-1, 3)
            first_field = sorted(h5_file.keys())[0]
            num_samples = h5_file[first_field].shape[0]
            self.ntrain = int(num_samples * train_ratio)

            # Set indices based on train or test
            if self.is_train:
                self.indices = np.arange(self.ntrain)
            else:
                self.indices = np.arange(self.ntrain, num_samples)
        self.fields = ["Vx", "Vy", "Vz", "pressure", "density"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        # Initialize data arrays for this sample only (much smaller memory footprint)
        a_data = np.zeros((self.grid_flat.shape[0], self.T_in * self.out_dim))
        u_data = np.zeros((self.grid_flat.shape[0], self.T_out * self.out_dim))
        # import pdb; pdb.set_trace()

        with h5py.File(self.data_path, "r") as h5_file:
            # Load input timesteps
            for t_in in range(self.T_in):
                for f_idx, field in enumerate(self.fields):
                    var_data = h5_file[field][
                        sample_idx, t_in, :: self.r1, :: self.r2, :: self.r3
                    ][: self.s1, : self.s2, : self.s3]
                    var_data_flat = var_data.reshape(-1)
                    a_data[:, t_in * self.out_dim + f_idx] = var_data_flat

            # Load output timesteps
            for t_out in range(self.T_out):
                for f_idx, field in enumerate(self.fields):
                    var_data = h5_file[field][
                        sample_idx,
                        self.T_in + t_out,
                        :: self.r1,
                        :: self.r2,
                        :: self.r3,
                    ][: self.s1, : self.s2, : self.s3]
                    var_data_flat = var_data.reshape(-1)
                    u_data[:, t_out * self.out_dim + f_idx] = var_data_flat

        # Convert to tensors
        a_data = torch.tensor(a_data, dtype=torch.float)
        u_data = torch.tensor(u_data, dtype=torch.float)

        return self.grid_flat, a_data, u_data


class cfd3d(object):
    def __init__(self, args, dist):
        self.data_path = args.data_path
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.downsamplez = args.downsamplez
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.rank = dist.rank

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        r3 = self.downsamplez
        s1 = int(((128 - 1) / r1) + 1)
        s2 = int(((128 - 1) / r2) + 1)
        s3 = int(((128 - 1) / r3) + 1)

        train_dataset = cfd_3d_dataset(
            self.data_path,
            self.downsamplex,
            self.downsampley,
            self.downsamplez,
            self.T_in,
            self.T_out,
            self.out_dim,
            is_train=True,
            train_ratio=self.train_ratio,
        )

        test_dataset = cfd_3d_dataset(
            self.data_path,
            self.downsamplex,
            self.downsampley,
            self.downsamplez,
            self.T_in,
            self.T_out,
            self.out_dim,
            is_train=False,
            train_ratio=self.train_ratio,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )
        if self.rank == 0:
            print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2, s3]


class HDF5ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.grid_size = datasets[0].grid_size if datasets else None

    def grid_size(self):
        return self._grid_size

    def future_window(self):
        return self.datasets[0].future_window

    def absmax_vel(self):
        return max(d.absmax_vel() for d in self.datasets)

    def absmax_temp(self):
        return max(d.absmax_temp() for d in self.datasets)

    def normalize_temp_(self, absmax_temp=None):
        if not absmax_temp:
            absmax_temp = self.absmax_temp()
        for d in self.datasets:
            d.normalize_temp_(absmax_temp)
        return absmax_temp

    def normalize_vel_(self, absmax_vel=None):
        if not absmax_vel:
            absmax_vel = self.absmax_vel()
        for d in self.datasets:
            d.normalize_vel_(absmax_vel)
        return absmax_vel

    def datum_dim(self):
        return self.datasets[0].datum_dim()


class HDF5Dataset(Dataset):
    def __init__(
        self,
        filename,
        steady_time,
        downsamplex=1,
        downsampley=1,
        transform=False,
        time_window=1,
        future_window=1,
        push_forward_steps=1,
    ):
        super().__init__()
        assert time_window > 0, "HDF5Dataset.__init__():time window should be positive"
        self.filename = filename
        self.steady_time = steady_time
        self.transform = transform
        self.time_window = time_window
        self.future_window = future_window
        self.push_forward_steps = push_forward_steps
        self.temp_scale = None
        self.vel_scale = None
        self.grid_size = None

        # 添加下采样参数
        self.downsamplex = downsamplex
        self.downsampley = downsampley
        # 计算下采样后的网格尺寸
        self.s1 = None
        self.s2 = None

        self.reset()

    def reset(self):
        self._data = {}
        with h5py.File(self.filename, "r") as f:
            first_temp = f["temperature"][0]
            H, W = first_temp.shape  # 保存网格尺寸
            # 计算下采样后的网格尺寸
            self.s1 = int((H - 1) / self.downsamplex) + 1
            self.s2 = int((W - 1) / self.downsampley) + 1

            def downsample_data(data):
                """应用下采样到数据"""
                # 选择时间步和空间维度
                return data[
                    self.steady_time :, :: self.downsamplex, :: self.downsampley
                ][:, : self.s1, : self.s2]
                # 读取并下采样数据

            self._data["temp"] = torch.nan_to_num(
                torch.from_numpy(downsample_data(f["temperature"][:]))
            )
            self._data["velx"] = torch.nan_to_num(
                torch.from_numpy(downsample_data(f["velx"][:]))
            )
            self._data["vely"] = torch.nan_to_num(
                torch.from_numpy(downsample_data(f["vely"][:]))
            )
            self._data["dfun"] = torch.nan_to_num(
                torch.from_numpy(downsample_data(f["dfun"][:]))
            )

            # 读取坐标数据（只取第一个时间步，假设网格不变）
            x_coords = f["x"][0][:: self.downsamplex, :: self.downsampley][
                : self.s1, : self.s2
            ]
            y_coords = f["y"][0][:: self.downsamplex, :: self.downsampley][
                : self.s1, : self.s2
            ]
            # 归一化网格坐标
            x_coords = x_coords / x_coords.max()
            y_coords = y_coords / y_coords.max()
            # 创建网格坐标张量（重复所有时间步）
            num_timesteps = self._data["temp"].shape[0]
            self._data["x"] = (
                torch.tensor(x_coords).unsqueeze(0).repeat(num_timesteps, 1, 1)
            )
            self._data["y"] = (
                torch.tensor(y_coords).unsqueeze(0).repeat(num_timesteps, 1, 1)
            )

        # 更新网格尺寸
        self.grid_size = (self.s1, self.s2)

        self._redim_temp(self.filename)
        if self.temp_scale and self.vel_scale:
            self.normalize_temp_(self.temp_scale)
            self.normalize_vel_(self.vel_scale)

    def datum_dim(self):
        return self._data["temp"].size()

    def _redim_temp(self, filename):
        r"""
        Each hdf5 file non-dimensionalizes temperature to the same range.
        If the wall temperature is varied across simulations, then the temperature
        must be re-dimensionalized, so it can be properly normalized across
        simulations.
        this is ONLY DONE WHEN THE FILENAME INCLUDES Twall-
        """
        filename = Path(filename).stem
        wall_temp = None
        TWALL = "Twall-"
        if TWALL in filename:
            self._data["temp"] *= int(filename[len(TWALL) :])
            print("wall temp", self._data["temp"].max())

    def absmax_temp(self):
        return self._data["temp"].abs().max()

    def absmax_vel(self):
        return max(self._data["velx"].abs().max(), self._data["vely"].abs().max())

    def normalize_temp_(self, scale):
        self._data["temp"] = 2 * (self._data["temp"] / scale) - 1
        self.temp_scale = scale

    def normalize_vel_(self, scale):
        for v in ("velx", "vely"):
            self._data[v] = self._data[v] / scale
        self.vel_scale = scale

    def get_x(self):
        return self._data["x"][self.time_window :]

    def get_dy(self):
        r"""dy is the grid spacing in the y direction."""
        return self._data["y"][0, 0, 0]

    def get_dfun(self):
        return self._data["dfun"][self.time_window :]

    def _get_temp(self, timestep):
        return self._data["temp"][timestep]

    def _get_vel_stack(self, timestep):
        return torch.stack(
            [
                self._data["velx"][timestep],
                self._data["vely"][timestep],
            ],
            dim=0,
        )

    def _get_coords(self, timestep):
        x = self._data["x"][timestep]
        x /= x.max()
        y = self._data["y"][timestep]
        y /= y.max()
        coords = torch.stack([x, y], dim=0)
        return coords

    def _get_dfun(self, timestep):
        vapor_mask = self._data["dfun"][timestep] > 0
        return vapor_mask.to(float) - 0.5

    def __len__(self):
        # len is the number of timesteps. Each prediction
        # requires time_window frames, so we can't predict for
        # the first few frames.
        # we may also predict several frames in the future, so we
        # can't include those in length
        return (
            self._data["temp"].size(0)
            - self.time_window
            - (self.future_window * self.push_forward_steps - 1)
        )

    def _transform(self, *args):
        if self.transform:
            if random.random() > 0.5:
                args = tuple([TF.hflip(arg) for arg in args])
        return args

    def __getitem__(self, timestep):
        assert False, "Not Implemented"


class TempInputDataset(HDF5Dataset):
    r"""
    This is a dataset for predicting only temperature. It assumes that
    velocities are known in every timestep. It also enables writing
    past predictions for temperature and using them to make future
    predictions.
    """

    def __init__(
        self,
        filename,
        steady_time,
        use_coords,
        downsamplex=1,
        downsampley=1,
        transform=False,
        time_window=1,
        future_window=1,
        push_forward_steps=1,
    ):
        super().__init__(
            filename,
            steady_time,
            downsamplex=downsamplex,
            downsampley=downsampley,
            transform=transform,
            time_window=time_window,
            future_window=future_window,
            push_forward_steps=push_forward_steps,
        )
        coords_dim = 2 if use_coords else 0
        self.in_channels = 3 * self.time_window + coords_dim + 2 * self.future_window
        self.out_channels = self.future_window

    def __getitem__(self, timestep):
        coords = self._get_coords(timestep)
        temps = torch.stack(
            [self._get_temp(timestep + k) for k in range(self.time_window)], dim=0
        )
        vel = torch.cat(
            [
                self._get_vel_stack(timestep + k)
                for k in range(self.time_window + self.future_window)
            ],
            dim=0,
        )
        base_time = timestep + self.time_window
        label = torch.stack(
            [self._get_temp(base_time + k) for k in range(self.future_window)], dim=0
        )
        return (coords, *self._transform(temps, vel, label))

    def write_temp(self, temp, timestep):
        if temp.dim() == 2:
            temp.unsqueeze_(-1)
        base_time = timestep + self.time_window
        self._data["temp"][base_time : base_time + self.future_window] = temp


class BubbleTemp(object):
    def __init__(self, args, dist):
        # 用 args.data_path 替换 config 的变量
        with hydra.initialize(config_path="../data/conf", version_base=None):
            # 使用默认配置文件名或根据 args 动态确定
            config_name = getattr(args, "config_name", "config")  # 默认 'config'
            cfg = hydra.compose(config_name=config_name)

        if hasattr(args, "data_path") and args.data_path is not None:
            cfg.data_base_dir = args.data_path
        self.loader_config = cfg

        self.data_path = args.data_path
        self.train_paths = self.loader_config.train_paths
        self.val_paths = self.loader_config.val_paths
        self.push_forward_steps = self.loader_config.push_forward_steps
        self.time_window = args.T_in
        self.future_window = args.T_out
        self.out_dim = args.out_dim
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.rank = dist.rank

    def get_loader(self):
        # 创建训练数据集列表
        train_datasets = []
        for file_path in self.train_paths:
            dataset = TempInputDataset(
                file_path,
                steady_time=self.loader_config.steady_time,
                use_coords=self.loader_config.use_coords,
                downsamplex=self.downsamplex,
                downsampley=self.downsampley,
                transform=self.loader_config.transform,
                time_window=self.time_window,
                future_window=self.future_window,
                push_forward_steps=self.loader_config.push_forward_steps,
            )
            train_datasets.append(dataset)

        # 创建验证数据集列表
        test_datasets = []
        for file_path in self.val_paths:
            dataset = TempInputDataset(
                file_path,
                steady_time=self.loader_config.steady_time,
                use_coords=self.loader_config.use_coords,
                downsamplex=self.downsamplex,
                downsampley=self.downsampley,
                time_window=self.time_window,
                future_window=self.future_window,
            )
            test_datasets.append(dataset)

        # 组合数据集
        train_dataset = HDF5ConcatDataset(train_datasets)
        test_dataset = HDF5ConcatDataset(test_datasets)

        # 使用与训练数据集相同的映射来对验证集进行归一化
        train_max_temp = train_dataset.normalize_temp_()
        train_max_vel = train_dataset.normalize_vel_()

        test_dataset.normalize_temp_(train_max_temp)
        test_dataset.normalize_vel_(train_max_vel)

        assert test_dataset.absmax_temp() <= 1.5
        assert test_dataset.absmax_vel() <= 1.5

        s1, s2 = train_dataset.grid_size

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        if self.rank == 0:
            print("Bubble dataset dataloading is ready.")
        return train_loader, test_loader, [s1, s2]


class TempVelDataset(HDF5Dataset):
    r"""
    This is a dataset for predicting both temperature and velocity.
    Velocities and temperatures are unknown. The model writes past
    predictions to reuse for future predictions.
    """

    def __init__(
        self,
        filename,
        steady_time,
        use_coords,
        downsamplex=1,
        downsampley=1,
        transform=False,
        time_window=1,
        future_window=1,
        push_forward_steps=1,
    ):
        # 传递下采样参数到父类
        super().__init__(
            filename,
            steady_time,
            downsamplex=downsamplex,
            downsampley=downsampley,
            transform=transform,
            time_window=time_window,
            future_window=future_window,
            push_forward_steps=push_forward_steps,
        )
        coords_dim = 2 if use_coords else 0
        self.temp_channels = self.time_window
        self.vel_channels = self.time_window * 2
        self.dfun_channels = self.time_window

        self.in_channels = (
            coords_dim + self.temp_channels + self.vel_channels + self.dfun_channels
        )
        self.out_channels = 3 * self.future_window

    def _get_timestep(self, timestep):
        r"""
        Get the window rooted at timestep.
        This includes the {timestep - self.time_window, ..., timestep - 1} as input
        and {timestep, ..., timestep + future_window - 1} as output
        """
        coords = self._get_coords(timestep)
        temp = torch.stack(
            [self._get_temp(timestep + k) for k in range(self.time_window)], dim=0
        )
        vel = torch.cat(
            [self._get_vel_stack(timestep + k) for k in range(self.time_window)], dim=0
        )
        dfun = torch.stack(
            [self._get_dfun(timestep + k) for k in range(self.time_window)], dim=0
        )

        base_time = timestep + self.time_window
        temp_label = torch.stack(
            [self._get_temp(base_time + k) for k in range(self.future_window)], dim=0
        )
        vel_label = torch.cat(
            [self._get_vel_stack(base_time + k) for k in range(self.future_window)],
            dim=0,
        )
        return self._transform(coords, temp, vel, dfun, temp_label, vel_label)

    def __getitem__(self, timestep):
        r"""
        Get the windows rooted at {timestep, timestep + self.future_window, ...}
        For each variable, the windows are concatenated into one tensor.
        """
        args = list(
            zip(
                *[
                    self._get_timestep(timestep + k * self.future_window)
                    for k in range(self.push_forward_steps)
                ]
            )
        )
        return tuple([torch.stack(arg, dim=0) for arg in args])

    def write_vel(self, vel, timestep):
        base_time = timestep + self.time_window
        self._data["velx"][base_time : base_time + self.future_window] = vel[0::2]
        self._data["vely"][base_time : base_time + self.future_window] = vel[1::2]

    def write_temp(self, temp, timestep):
        if temp.dim() == 2:
            temp.unsqueeze_(-1)
        base_time = timestep + self.time_window
        self._data["temp"][base_time : base_time + self.future_window] = temp


class BubbleTempVel(object):
    def __init__(self, args, dist):
        # 用 args.data_path 替换 config 的变量
        with hydra.initialize(config_path="../data/conf", version_base=None):
            # 使用默认配置文件名或根据 args 动态确定
            config_name = getattr(args, "config_name", "config")  # 默认 'config'
            cfg = hydra.compose(config_name=config_name)

        if hasattr(args, "data_path") and args.data_path is not None:
            cfg.data_base_dir = args.data_path
        self.loader_config = cfg

        self.data_path = args.data_path
        self.train_paths = self.loader_config.train_paths
        self.val_paths = self.loader_config.val_paths
        self.push_forward_steps = self.loader_config.push_forward_steps
        self.time_window = args.T_in
        self.future_window = args.T_out
        self.out_dim = args.out_dim
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.rank = dist.rank

    def get_loader(self):
        # 创建训练数据集列表
        train_datasets = []
        for file_path in self.train_paths:
            dataset = TempVelDataset(
                file_path,
                steady_time=self.loader_config.steady_time,
                use_coords=self.loader_config.use_coords,
                downsamplex=self.downsamplex,
                downsampley=self.downsampley,
                transform=self.loader_config.transform,
                time_window=self.time_window,
                future_window=self.future_window,
                push_forward_steps=self.loader_config.push_forward_steps,
            )
            train_datasets.append(dataset)

        # 创建验证数据集列表
        test_datasets = []
        for file_path in self.val_paths:
            dataset = TempVelDataset(
                file_path,
                steady_time=self.loader_config.steady_time,
                use_coords=self.loader_config.use_coords,
                downsamplex=self.downsamplex,
                downsampley=self.downsampley,
                time_window=self.time_window,
                future_window=self.future_window,
            )
            test_datasets.append(dataset)

        # 组合数据集
        train_dataset = HDF5ConcatDataset(train_datasets)
        test_dataset = HDF5ConcatDataset(test_datasets)

        # 使用与训练数据集相同的映射来对验证集进行归一化
        train_max_temp = train_dataset.normalize_temp_()
        train_max_vel = train_dataset.normalize_vel_()

        test_dataset.normalize_temp_(train_max_temp)
        test_dataset.normalize_vel_(train_max_vel)

        assert test_dataset.absmax_temp() <= 1.5
        assert test_dataset.absmax_vel() <= 1.5

        s1, s2 = train_dataset.grid_size

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        if self.rank == 0:
            print("Bubble dataset dataloading is ready.")
        return train_loader, test_loader, [s1, s2]
