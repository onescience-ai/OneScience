# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import shutil

import torch


class ESMStructuralSplitDataset(torch.utils.data.Dataset):
    """
    Structural Split Dataset as described in section A.10 of the supplement of our paper.
    https://doi.org/10.1101/622803
    """

    base_folder = "structural-data"
    file_list = [
        #  url  tar filename   filename      MD5 Hash
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz",
            "splits.tar.gz",
            "splits",
            "456fe1c7f22c9d3d8dfe9735da52411d",
        ),
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz",
            "pkl.tar.gz",
            "pkl",
            "644ea91e56066c750cd50101d390f5db",
        ),
    ]

    def __init__(
        self,
        split_level,
        cv_partition,
        split,
        root_path=os.path.expanduser("~/.cache/torch/data/esm"),
        download=False,
    ):
        super().__init__()
        assert split in [
            "train",
            "valid",
        ], "train_valid must be 'train' or 'valid'"
        self.root_path = root_path
        self.base_path = os.path.join(self.root_path, self.base_folder)

        # check if root path has what you need or else download it
        if download:
            self.download()

        self.split_file = os.path.join(
            self.base_path, "splits", split_level, cv_partition, f"{split}.txt"
        )
        self.pkl_dir = os.path.join(self.base_path, "pkl")
        self.names = []
        with open(self.split_file) as f:
            self.names = f.read().splitlines()

    def __len__(self):
        return len(self.names)

    def _check_exists(self) -> bool:
        for (_, _, filename, _) in self.file_list:
            fpath = os.path.join(self.base_path, filename)
            if not os.path.exists(fpath) or not os.path.isdir(fpath):
                return False
        return True

    def download(self):

        if self._check_exists():
            print("Files already downloaded and verified")
            return

        from torchvision.datasets.utils import download_url

        for url, tar_filename, filename, md5_hash in self.file_list:
            download_path = os.path.join(self.base_path, tar_filename)
            download_url(url=url, root=self.base_path, filename=tar_filename, md5=md5_hash)
            shutil.unpack_archive(download_path, self.base_path)

    def __getitem__(self, idx):
        """
        Returns a dict with the following entires
         - seq : Str (domain sequence)
         - ssp : Str (SSP labels)
         - dist : np.array (distance map)
         - coords : np.array (3D coordinates)
        """
        name = self.names[idx]
        pkl_fname = os.path.join(self.pkl_dir, name[1:3], f"{name}.pkl")
        with open(pkl_fname, "rb") as f:
            obj = pickle.load(f)
        return obj
