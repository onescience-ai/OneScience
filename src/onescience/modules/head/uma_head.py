"""UMA output heads and dataset-specific head wrappers."""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn

from onescience.models.UMA.base import HeadInterface
from onescience.modules.block.uma_mole_block import MOLEGlobals
from onescience.modules.func_utils.uma_irreps import cg_change_mat, irreps_sum
from onescience.modules.func_utils.uma_mole_utils import (
    recursive_replace_all_linear,
    replace_linear_with_MOLE,
)
from onescience.modules.layer.uma_so3_layers import SO3_Linear
from onescience.utils.uma.common import gp_utils
from onescience.utils.uma.common.registry import registry
from onescience.utils.uma.common.utils import conditional_grad


class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, prefix=None, wrap_property=True):
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces
        self.prefix = prefix
        self.wrap_property = wrap_property

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        # this is required for gradient-based forces/stress in this head
        backbone.direct_forces = False
        assert (
            not backbone.direct_forces
        ), "EFS head is only used for gradient-based forces/stress."

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.prefix:
            energy_key = f"{self.prefix}_energy"
            forces_key = f"{self.prefix}_forces"
            stress_key = f"{self.prefix}_stress"
        else:
            energy_key = "energy"
            forces_key = "forces"
            stress_key = "stress"

        outputs = {}
        _input = emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)
        node_energy = _output.view(-1, 1, 1)
        energy_part = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy_part.index_add_(0, data["batch"], node_energy.view(-1))

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy_part.sum()],
                [data["pos_original"], emb["displacement"]],
                create_graph=self.training,
            )
            if gp_utils.initialized():
                grads = (
                    gp_utils.reduce_from_model_parallel_region(grads[0]),
                    gp_utils.reduce_from_model_parallel_region(grads[1]),
                )

            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            stress = stress.view(-1, 9)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
            data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy_part.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            if gp_utils.initialized():
                forces = gp_utils.reduce_from_model_parallel_region(forces)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        return outputs


class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        energy_part = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy_part.index_add_(0, data_dict["batch"], node_energy.view(-1))
        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        if self.reduce == "sum":
            return {"energy": energy}
        if self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        raise ValueError(f"reduce can only be sum or mean, user provided: {self.reduce}")


class Linear_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        energy_part = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy_part.index_add_(0, data_dict["batch"], node_energy.view(-1))

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        if self.reduce == "sum":
            return {"energy": energy}
        if self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        raise ValueError(f"reduce can only be sum or mean, user provided: {self.reduce}")


class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
        return {"forces": forces}


def compose_tensor(
    trace: torch.Tensor,
    l2_symmetric: torch.Tensor,
) -> torch.Tensor:
    if trace.shape[1] != 1:
        raise ValueError("batch of traces must be shape (batch size, 1)")

    if l2_symmetric.shape[1] != 5:
        raise ValueError("batch of l2_symmetric tensors must be shape (batch size, 5)")

    if trace.shape[0] != l2_symmetric.shape[0]:
        raise ValueError(
            "Shape missmatch between trace and l2_symmetric parts. The first dimension is the batch dimension"
        )

    batch_size = trace.shape[0]
    decomposed_preds = torch.zeros(
        batch_size, irreps_sum(2), device=trace.device
    )  # rank 2
    decomposed_preds[:, : irreps_sum(0)] = trace
    decomposed_preds[:, irreps_sum(1) : irreps_sum(2)] = l2_symmetric

    r2_tensor = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(2, device=trace.device),
        decomposed_preds,
    )
    return r2_tensor


class MLP_Stress_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "mean"):
        super().__init__()
        self.reduce = reduce
        assert reduce in ["sum", "mean"]
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.scalar_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        self.l2_linear = SO3_Linear(backbone.sphere_channels, 1, lmax=2)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        node_scalar = self.scalar_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        iso_stress = torch.zeros(
            len(data_dict["natoms"]),
            device=node_scalar.device,
            dtype=node_scalar.dtype,
        )
        iso_stress.index_add_(0, data_dict["batch"], node_scalar.view(-1))

        if gp_utils.initialized():
            raise NotImplementedError("This code hasn't been tested yet.")

        if self.reduce == "mean":
            iso_stress /= data_dict["natoms"]

        node_l2 = self.l2_linear(emb["node_embedding"].narrow(1, 0, 9))
        node_l2 = node_l2.narrow(1, 4, 5)
        node_l2 = node_l2.view(-1, 5).contiguous()

        aniso_stress = torch.zeros(
            (len(data_dict["natoms"]), 5),
            device=node_l2.device,
            dtype=node_l2.dtype,
        )
        aniso_stress.index_add_(0, data_dict["batch"], node_l2)
        if gp_utils.initialized():
            raise NotImplementedError("This code hasn't been tested yet.")

        if self.reduce == "mean":
            aniso_stress /= data_dict["natoms"].unsqueeze(1)

        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress)

        return {"stress": stress}


class DatasetSpecificMoEWrapper(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        dataset_names,
        head_cls,
        wrap_property=True,
        head_kwargs=None,
    ):
        super().__init__()
        if head_kwargs is None:
            head_kwargs = {}
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.wrap_property = wrap_property

        self.dataset_names = sorted(dataset_names)
        self.dataset_name_to_exp = {
            value: idx for idx, value in enumerate(self.dataset_names)
        }
        self.head = registry.get_model_class(head_cls)(backbone, **head_kwargs)
        self.global_mole_tensors = MOLEGlobals(
            expert_mixing_coefficients=None, mole_sizes=None
        )
        replacement_factory = functools.partial(
            replace_linear_with_MOLE,
            num_experts=len(self.dataset_names),
            global_mole_tensors=self.global_mole_tensors,
            mole_layer_type="pytorch",
            cache=None,
        )
        recursive_replace_all_linear(self.head, replacement_factory)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.global_mole_tensors.mole_sizes = torch.zeros(
            data.natoms.shape[0], dtype=torch.int, device=emb["batch"].device
        ).scatter(0, emb["batch"], 1, reduce="add")
        self.global_mole_tensors.natoms = emb["batch"].shape[0]
        data_batch_full = data.batch_full.cpu()

        self.global_mole_tensors.expert_mixing_coefficients = (
            torch.zeros(
                data.natoms.shape[0],
                len(self.dataset_name_to_exp),
                dtype=data.pos.dtype,
            )
            .scatter(
                1,
                torch.tensor(
                    [
                        self.dataset_name_to_exp[dataset_name]
                        for dataset_name in data.dataset
                    ],
                ).unsqueeze(1),
                1.0,
            )
            .to(data.pos.device)
        )

        head_output = self.head(data, emb)

        np_dataset_names = np.array(data.dataset)
        full_output = {}
        for dataset_name in self.dataset_names:
            dataset_mask = np_dataset_names == dataset_name
            for key, mole_output_tensor in head_output.items():
                output_tensor = mole_output_tensor.new_zeros(mole_output_tensor.shape)
                if dataset_mask.any():
                    if output_tensor.shape[0] == dataset_mask.shape[0]:
                        output_tensor[dataset_mask] = mole_output_tensor[dataset_mask]
                    else:
                        atoms_mask = torch.isin(
                            data_batch_full,
                            torch.where(torch.from_numpy(dataset_mask))[0],
                        )
                        output_tensor[atoms_mask] = mole_output_tensor[atoms_mask]
                full_output[f"{dataset_name}_{key}"] = (
                    {key: output_tensor} if self.wrap_property else output_tensor
                )

        return full_output


class DatasetSpecificSingleHeadWrapper(nn.Module, HeadInterface):
    def __init__(
        self, backbone, dataset_names, head_cls, wrap_property=True, head_kwargs=None
    ):
        super().__init__()
        if head_kwargs is None:
            head_kwargs = {}
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.wrap_property = wrap_property

        self.dataset_names = sorted(dataset_names)
        self.head = registry.get_model_class(head_cls)(backbone, **head_kwargs)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data_batch_full = data.batch_full.cpu()
        head_output = self.head(data, emb)

        assert set(data.dataset) <= set(
            self.dataset_names
        ), f"Input dataset names: {set(data.dataset)} must be a strict subset of model's valid datset names: {set(self.dataset_names)} "
        np_dataset_names = np.array(data.dataset)

        full_output = {}
        for dataset_name in self.dataset_names:
            dataset_mask = np_dataset_names == dataset_name
            for key, head_output_tensor in head_output.items():
                output_tensor = head_output_tensor.new_zeros(head_output_tensor.shape)
                if dataset_mask.any():
                    if output_tensor.shape[0] == dataset_mask.shape[0]:
                        output_tensor[dataset_mask] = head_output_tensor[dataset_mask]
                    else:
                        atoms_mask = torch.isin(
                            data_batch_full,
                            torch.where(torch.from_numpy(dataset_mask))[0],
                        )
                        output_tensor[atoms_mask] = head_output_tensor[atoms_mask]
                full_output[f"{dataset_name}_{key}"] = (
                    {key: output_tensor} if self.wrap_property else output_tensor
                )

        return full_output


__all__ = [
    "DatasetSpecificMoEWrapper",
    "DatasetSpecificSingleHeadWrapper",
    "Linear_Energy_Head",
    "Linear_Force_Head",
    "MLP_EFS_Head",
    "MLP_Energy_Head",
    "MLP_Stress_Head",
    "compose_tensor",
]
