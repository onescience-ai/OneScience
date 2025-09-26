import math as mt
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class UNODatasetSingle(Dataset):
    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        test_ratio=0.1,
        num_samples_max=-1,
    ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        root_path = os.path.abspath(saved_folder + filename)
        assert filename[-2:] != "h5", "HDF5 data is assumed!!"

        with h5py.File(root_path, "r") as f:
            keys = list(f.keys())
            keys.sort()
            if "tensor" not in keys:
                # batch, time, x,...
                _data = np.array(
                    f["density"], dtype=np.float32)
                idx_cfd = _data.shape
                if len(idx_cfd) == 3:  # 1D
                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            mt.ceil(
                                idx_cfd[1] / reduced_resolution_t),
                            3,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data[:, :, :], (0, 2, 1))
                    # batch, x, t, ch
                    self.data[..., 0] = _data
                    # pressure
                    _data = np.array(
                        f["pressure"], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data[:, :, :], (0, 2, 1))
                    # batch, x, t, ch
                    self.data[..., 1] = _data
                    # Vx
                    # batch, time, x,...
                    _data = np.array(
                        f["Vx"], dtype=np.float32)
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data[:, :, :], (0, 2, 1))
                    # batch, x, t, ch
                    self.data[..., 2] = _data

                    self.grid = np.array(
                        f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(
                        self.grid[::reduced_resolution], dtype=torch.float
                    ).unsqueeze(-1)
                    print(self.data.shape)
                if len(idx_cfd) == 4:  # 2D
                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            idx_cfd[3] // reduced_resolution,
                            mt.ceil(
                                idx_cfd[1] / reduced_resolution_t),
                            4,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 1))
                    # batch, x, t, ch
                    self.data[..., 0] = _data
                    # pressure
                    _data = np.array(
                        f["pressure"], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 1))
                    # batch, x, t, ch
                    self.data[..., 1] = _data
                    # Vx
                    # batch, time, x,...
                    _data = np.array(
                        f["Vx"], dtype=np.float32)
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 1))
                    # batch, x, t, ch
                    self.data[..., 2] = _data
                    # Vy
                    # batch, time, x,...
                    _data = np.array(
                        f["Vy"], dtype=np.float32)
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 1))
                    # batch, x, t, ch
                    self.data[..., 3] = _data

                    x = np.array(
                        f["x-coordinate"], dtype=np.float32)
                    y = np.array(
                        f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[
                        ::reduced_resolution, ::reduced_resolution
                    ]

                if len(idx_cfd) == 5:  # 3D
                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            idx_cfd[3] // reduced_resolution,
                            idx_cfd[4] // reduced_resolution,
                            mt.ceil(
                                idx_cfd[1] / reduced_resolution_t),
                            5,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 4, 1))
                    # batch, x, t, ch
                    self.data[..., 0] = _data
                    # pressure
                    _data = np.array(
                        f["pressure"], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 4, 1))
                    # batch, x, t, ch
                    self.data[..., 1] = _data
                    # Vx
                    # batch, time, x,...
                    _data = np.array(
                        f["Vx"], dtype=np.float32)
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 4, 1))
                    # batch, x, t, ch
                    self.data[..., 2] = _data
                    # Vy
                    # batch, time, x,...
                    _data = np.array(
                        f["Vy"], dtype=np.float32)
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 4, 1))
                    # batch, x, t, ch
                    self.data[..., 3] = _data
                    # Vz
                    # batch, time, x,...
                    _data = np.array(
                        f["Vz"], dtype=np.float32)
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data, (0, 2, 3, 4, 1))
                    # batch, x, t, ch
                    self.data[..., 4] = _data

                    x = np.array(
                        f["x-coordinate"], dtype=np.float32)
                    y = np.array(
                        f["y-coordinate"], dtype=np.float32)
                    z = np.array(
                        f["z-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    z = torch.tensor(z, dtype=torch.float)
                    X, Y, Z = torch.meshgrid(x, y, z)
                    self.grid = torch.stack((X, Y, Z), axis=-1)[
                        ::reduced_resolution, ::reduced_resolution, ::reduced_resolution
                    ]

            else:  # scalar equations
                # data dim = [t, x1, ..., xd, v]
                # batch, time, x,...
                _data = np.array(
                    f["tensor"], dtype=np.float32)
                if len(_data.shape) == 3:  # 1D
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    # convert to [x1, ..., xd, t, v]
                    _data = np.transpose(
                        _data[:, :, :], (0, 2, 1))
                    # batch, x, t, ch
                    self.data = _data[:, :, :, None]

                    self.grid = np.array(
                        f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(
                        self.grid[::reduced_resolution], dtype=torch.float
                    ).unsqueeze(-1)
                elif len(_data.shape) == 4:
                    if "nu" in f.keys():  # 2D Darcy flow
                        # u: label
                        _data = _data[
                            ::reduced_batch,
                            :,
                            ::reduced_resolution,
                            ::reduced_resolution,
                        ]
                        # convert to [x1, ..., xd, t, v]
                        _data = np.transpose(
                            _data[:, :, :, :], (0, 2, 3, 1))
                        # if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(
                            f["nu"], dtype=np.float32
                        )  # batch, time, x,...
                        _data = _data[
                            ::reduced_batch,
                            None,
                            ::reduced_resolution,
                            ::reduced_resolution,
                        ]
                        # convert to [x1, ..., xd, t, v]
                        _data = np.transpose(
                            _data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate(
                            [_data, self.data], axis=-1)
                        # batch, x, y, t, ch
                        self.data = self.data[:,
                                              :, :, :, None]
                    else:
                        # label: (num_sample, t, x1, x2)
                        _data = _data[
                            ::reduced_batch,
                            :,
                            ::reduced_resolution,
                            ::reduced_resolution,
                        ]
                        # convert to (num_sample, x1, ..., xd, t)
                        _data = np.transpose(
                            _data[:, :, :, :], (0, 2, 3, 1))
                        self.data = _data[:, :, :, :, None]

                    x = np.array(
                        f["x-coordinate"], dtype=np.float32)
                    y = np.array(
                        f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[
                        ::reduced_resolution, ::reduced_resolution
                    ]
                # TODO 3D
                else:
                    pass

        if num_samples_max > 0:
            num_samples_max = min(
                num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx, ..., : self.initial_step, :], self.data[idx], self.grid


class UNODatasetMult(Dataset):
    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        test_ratio=0.1,
    ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        # Define path to files
        self.file_path = os.path.abspath(
            saved_folder + filename)

        # Extract list of seeds
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())
        data_list = data_list[::reduced_batch]

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # Open file and read data
        with h5py.File(self.file_path, "r") as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype="f")
            if len(data.shape) == 3:  # 1D
                data = data[:: self.reduced_resolution_t,
                            :: self.reduced_resolution, :]
            elif len(data.shape) == 4:  # 2D
                data = data[
                    :: self.reduced_resolution_t,
                    :: self.reduced_resolution,
                    :: self.reduced_resolution,
                    :,
                ]
            else:  # TODO 3D
                data = data[
                    :: self.reduced_resolution_t,
                    :: self.reduced_resolution,
                    :: self.reduced_resolution,
                    :: self.reduced_resolution,
                    :,
                ]

            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(
                range(1, len(data.shape) - 1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            if "global_maximums" in seed_group.keys():  # maxwell
                global_maximums = np.array(
                    seed_group["global_maximums"], dtype="f")
                global_maximums = torch.tensor(
                    global_maximums, dtype=torch.float)
                return data[..., : self.initial_step, :], data, global_maximums
            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(
                    seed_group["grid"]["x"], dtype="f")
                grid = torch.tensor(
                    grid[:: self.reduced_resolution], dtype=torch.float
                ).unsqueeze(-1)
            elif dim == 2:
                x = np.array(
                    seed_group["grid"]["x"], dtype="f")
                y = np.array(
                    seed_group["grid"]["y"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y)
                grid = torch.stack((X, Y), axis=-1)[
                    :: self.reduced_resolution, :: self.reduced_resolution
                ]
            elif dim == 3:
                x = np.array(
                    seed_group["grid"]["x"], dtype="f")
                y = np.array(
                    seed_group["grid"]["y"], dtype="f")
                z = np.array(
                    seed_group["grid"]["z"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z)
                grid = torch.stack((X, Y, Z), axis=-1)[
                    :: self.reduced_resolution,
                    :: self.reduced_resolution,
                    :: self.reduced_resolution,
                ]

        return data[..., : self.initial_step, :], data, grid


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
