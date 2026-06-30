import math
from dataclasses import dataclass

import numpy as np
import torch

from onescience.models.meta import ModelMetaData
from onescience.modules.module import Module
from onescience.modules import (
    OneEmbedding,
    OneFuser,
    OneRecovery,
    OneSample,
)

from onescience.distributed.megatron.training import get_args
from onescience.distributed.megatron.training.arguments import core_transformer_config_from_args
from onescience.distributed.megatron.core.tensor_parallel import checkpoint


@dataclass
class MetaData(ModelMetaData):
    name: str = "Pangu"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Pangu(Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        config = None
    ):
        super().__init__(meta=MetaData())
        self.l1d = 2
        self.l2d = 6
        self.l3d = 6
        self.l4d = 2
        # self.l1d = 4
        # self.l2d = 8
        # self.l3d = 8
        # self.l4d = 4
        self.pre_process = None
        self.share_embeddings_and_output_weights = None
        args = get_args()
        self.config = core_transformer_config_from_args(args)
        drop_path = np.linspace(0, 0.2, self.l1d + self.l2d + self.l3d + self.l4d).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        
        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size[1:],
            embed_dim=embed_dim,
            Variables=7
        )
        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding",
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            Variables=5,
            embed_dim=embed_dim,
        )
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.layer1 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=self.l1d,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:self.l1d],
            config = config
        )

        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        self.downsample = OneSample(
            style="PanguDownSample",
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
            in_dim=embed_dim,
        )
        self.layer2 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=self.l2d,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[-self.l2d:],
            config = config
        )
        self.layer3 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=self.l3d,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[-self.l3d:],
            config = config
        )
        self.upsample = OneSample(
            style="PanguUpSample",
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
            input_resolution=patched_inp_shape_downsample,
            output_resolution=patched_inp_shape
        )
        self.layer4 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=self.l4d,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:self.l4d],
            config = config
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=img_size,
            patch_size=patch_size[1:],
            in_chans=2 * embed_dim,
            out_chans=4,
        )
        self.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            in_chans=2 * embed_dim,
            out_chans=5,
        )

        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def prepare_input(self, surface, surface_mask, upper_air):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        upper_air = upper_air.reshape(
            upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4]
        )
        surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
        return torch.concat([surface, surface_mask, upper_air], dim=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        surface = x[:, :7, :, :]
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = x.contiguous()
        x = self.layer1(x)

        skip = x
        
        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]
			
        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air


class Pangu_stage0(Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        config = None
    ):
        super().__init__(meta=MetaData())
        # self.l1d = 2
        # self.l2d = 6
        # self.l3d = 6
        # self.l4d = 2
        # self.l1d = 6
        # self.l2d = 10
        # self.l3d = 10
        # self.l4d = 6
        self.l1d = 4
        self.l2d = 8
        self.l3d = 8
        self.l4d = 4
        self.pre_process = None
        self.share_embeddings_and_output_weights = None
        args = get_args()
        self.config = core_transformer_config_from_args(args)
        
        drop_path = np.linspace(0, 0.2, self.l1d + self.l2d + self.l3d + self.l4d).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        
        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size[1:],
            embed_dim=embed_dim,
            Variables=7,
        )
        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding",
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            Variables=5,
            embed_dim=embed_dim,
        )
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.layer1 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=self.l1d,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:self.l1d],
            config = config
        )

        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        self.downsample = OneSample(
            style="PanguDownSample",
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
            in_dim=embed_dim,
        )
        self.layer2 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=self.l2d,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[-self.l2d:],
			config = config
        )
        
    def prepare_input(self, surface, surface_mask, upper_air):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        upper_air = upper_air.reshape(
            upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4]
        )
        surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
        return torch.concat([surface, surface_mask, upper_air], dim=1)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        surface = x[:, :7, :, :]
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        
        x = checkpoint(self.layer1,False,x)
        skip = x

        x = self.downsample(x)
        
        x = checkpoint(self.layer2,False,x)
        
        x = x.contiguous()
        skip = skip.contiguous()

        return (x, skip)


class Pangu_stage1(Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        config = None
    ):
        super().__init__(meta=MetaData())
        # self.l1d = 2
        # self.l2d = 6
        # self.l3d = 6
        # self.l4d = 2
        # self.l1d = 6
        # self.l2d = 10
        # self.l3d = 10
        # self.l4d = 6
        self.l1d = 4
        self.l2d = 8
        self.l3d = 8
        self.l4d = 4
        args = get_args()
        self.config = core_transformer_config_from_args(args)
        self.pre_process = None
        self.share_embeddings_and_output_weights = None
        
        drop_path = np.linspace(0, 0.2, self.l1d + self.l2d + self.l3d + self.l4d).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)

        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size[1:],
            embed_dim=embed_dim,
            Variables=7,
        )
        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding",
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            Variables=5,
            embed_dim=embed_dim,
        )
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        self.downsample = OneSample(
            style="PanguDownSample",
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
            in_dim=embed_dim,
        )

        self.layer3 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=self.l3d,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[-self.l3d:],
            config = config
        )
        self.upsample = OneSample(
            style="PanguUpSample",
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
            input_resolution=patched_inp_shape_downsample,
            output_resolution=patched_inp_shape
        )

        self.pl = 8
        self.lat_tokens = math.ceil(img_size[0] / patch_size[1])
        self.lon_tokens = math.ceil(img_size[1] / patch_size[2])

        self.layer4 = OneFuser(
            style="PanguDistributedFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=self.l4d,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:self.l4d],
            config = config
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.

        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=img_size,
            patch_size=patch_size[1:],
            in_chans=2 * embed_dim,
            out_chans=4,
        )
        self.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            in_chans=2 * embed_dim,
            out_chans=5,
        )

        self.input_tensor = None

    def prepare_input(self, surface, surface_mask, upper_air):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        upper_air = upper_air.reshape(
            upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4]
        )
        surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
        return torch.concat([surface, surface_mask, upper_air], dim=1)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]

        surface = x[:, :7, :, :]
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        skip = x

		x = self.downsample(x)
        x = self.layer2(x)
        """
        if self.input_tensor != None:
            x = self.input_tensor
        
        x, skip = x
        x = x.contiguous()
        skip = skip.contiguous()
        
        x = checkpoint(self.layer3,False,x)
        x = self.upsample(x)

        B = x.shape[0]
        Pl = self.pl
        Lat = self.lat_tokens
        Lon = self.lon_tokens
        
        x = checkpoint(self.layer4,False,x)

        output = torch.concat([x, skip], dim=-1)

        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air
