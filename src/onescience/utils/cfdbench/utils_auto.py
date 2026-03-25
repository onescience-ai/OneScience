from typing import Tuple

from onescience.models.cfdbench.resnet import ResNet
from onescience.models.cfdbench.unet import UNet
from onescience.models.cfdbench.base_model import AutoCfdModel
from onescience.models.cfdbench.auto_deeponet import AutoDeepONet
from onescience.models.cfdbench.auto_edeeponet import AutoEDeepONet
from onescience.models.cfdbench.auto_deeponet_cnn import AutoDeepONetCnn
from onescience.models.cfdbench.fno.fno2d import Fno2d
from onescience.models.cfdbench.auto_ffn import AutoFfn
from onescience.models.cfdbench.loss import loss_name_to_fn
import torch

def get_input_shapes(data_name: str, num_rows: int, num_cols: int) -> Tuple[int, int, int]:
    """
    根据数据集名称和基础行列数，计算实际的模型输入尺寸和物理参数数量。
    
    Args:
        data_name: 数据集名称 (e.g., "tube_prop", "cavity_geo")
        num_rows: 基础行数 (通常配置为 64)
        num_cols: 基础列数 (通常配置为 64)
    
    Returns:
        (n_rows, n_cols, n_case_params)
    """
    # 1. 计算网格尺寸 (Rows, Cols)
    # Tube, Dam, Cylinder 需要处理边界 Padding
    if any(x in data_name for x in ["tube", "dam", "cylinder"]):
        n_rows = num_rows + 2  # Top and bottom boundaries
        n_cols = num_cols + 1  # Left boundary
    elif "cavity" in data_name:
        n_rows = num_rows
        n_cols = num_cols
    else:
        raise ValueError(f"Unknown dataset name for shape calculation: {data_name}")

    # 2. 计算物理参数数量 (Case Params)
    if "cylinder" in data_name:
        # vel_in, density, viscosity, height, width, radius, center_x, center_y
        n_case_params = 8
    elif any(x in data_name for x in ["cavity", "tube", "dam"]):
        # vel_in, density, viscosity, height, width
        n_case_params = 5  # physical properties
    else:
        raise ValueError(f"Unknown dataset name for params calculation: {data_name}")

    return n_rows, n_cols, n_case_params



def init_model(cfg) -> torch.nn.Module:
    """
    初始化自回归模型
    Args:
        cfg: 完整的 YParams 配置对象 (包含 model, datapipe, training)
    """
    model_cfg = cfg.model
    data_cfg = cfg.datapipe.data
    source_cfg = cfg.datapipe.source
    train_cfg = cfg.training
    loss_fn = loss_name_to_fn(train_cfg.loss_name)
    
    n_rows, n_cols, n_case_params = get_input_shapes(
        data_name=source_cfg.data_name,
        num_rows=data_cfg.num_rows,
        num_cols=data_cfg.num_cols
    )
    
    if model_cfg.name == "auto_ffn":
        model = AutoFfn(
            input_field_dim=n_rows * n_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
            width=model_cfg.autoffn_width,
            depth=model_cfg.autoffn_depth,
        )
    elif model_cfg.name == "auto_deeponet":
        branch_dim = n_cols * n_rows + n_case_params
        model = AutoDeepONet(
            branch_dim=branch_dim,
            trunk_dim=2, # (x, y)
            loss_fn=loss_fn,
            width=model_cfg.deeponet_width,
            trunk_depth=model_cfg.trunk_depth,
            branch_depth=model_cfg.branch_depth,
            act_name=model_cfg.act_fn,
        )
    elif model_cfg.name == "auto_edeeponet":
        model = AutoEDeepONet(
            dim_branch1=n_rows * n_cols,
            dim_branch2=n_case_params,
            trunk_dim=2, # (x, y)
            loss_fn=loss_fn,
            width=model_cfg.autoedeeponet_width,
            trunk_depth=model_cfg.autoedeeponet_depth,
            branch_depth=model_cfg.autoedeeponet_depth,
            act_name=model_cfg.autoedeeponet_act_fn,
        )
    elif model_cfg.name == "auto_deeponet_cnn":
        model = AutoDeepONetCnn(
            in_chan=model_cfg.in_chan,
            height=n_rows,
            width=n_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
        )
    elif model_cfg.name == "resnet":
        # 假设 ResNet 类可用
        model = ResNet(
            in_chan=model_cfg.in_chan,
            out_chan=model_cfg.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            hidden_chan=model_cfg.resnet_hidden_chan,
            num_blocks=model_cfg.resnet_depth,
            kernel_size=model_cfg.resnet_kernel_size,
            padding=model_cfg.resnet_padding,
        )
    elif model_cfg.name == "unet":
        # 假设 UNet 类可用
        model = UNet(
            in_chan=model_cfg.in_chan,
            out_chan=model_cfg.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            insert_case_params_at=model_cfg.unet_insert_case_params_at,
            dim=model_cfg.unet_dim,
        )
    elif model_cfg.name == "fno":
        model = Fno2d(
            in_chan=model_cfg.in_chan,
            out_chan=model_cfg.out_chan,
            n_case_params=n_case_params,
            loss_fn=loss_fn,
            num_layers=model_cfg.fno_depth,
            hidden_dim=model_cfg.fno_hidden_dim,
            modes1=model_cfg.fno_modes_x,
            modes2=model_cfg.fno_modes_y,
        )
    else:
        raise ValueError(f"Invalid model name: {model_cfg.name}")
    
    return model.cuda()

