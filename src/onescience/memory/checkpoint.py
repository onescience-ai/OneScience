from contextlib import contextmanager
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint


class part(torch.nn.Module):
    """
    2025-01-09 wangxian
    这个类主要是为了避免pytorch的类别检查，为了兼容原生pytorch而设计
    """

    def __init__(self, ch, ori):
        super().__init__()
        self.ch = ch  # checkpoint 函数
        self.ori = ori  # 原始子模块

    def __call__(self, *args, **kwargs):
        f = partial(self.ch, self.ori, use_reentrant=True)
        return f(*args, **kwargs)


@contextmanager
def replace_function(module, replace_layers_list, ddp_flag=False):
    """
    2025-01-09 wangxian
    这个函数可以使得任意pytorch模型中的模块被替换为checkpoint函数，从而实现checkpoint的功能
    以下给出一个案例，替代模型训练中的forward过程

    example:
    with replace_function(my_model, ['layer1','layer2','layer3','layer4'],dist.world_size > 1):
        outpred_surface, outpred_upper_air = my_model(invar)


    模型前向被替换为上下文的包裹函数
    其中replace_layers_list是需要被替换的nn.module子类，注册在模型中，ddp_flag代表是否使用分布式训练，默认是false

    2025-07-11 songzhaolu
    改进：支持多级路径的子模块替换：
        replace_layers_list = ['layer',
                               'blocks.0',
                               'blocks.0.Attn', ...]
        以下面这个模型的参数为例，可以使用blocks.0和blocks.0.Attn等访问ModuleList中的nn.module子类：
        (blocks): ModuleList(
        (0-6): 7 x Transolver_block(
            (ln_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (Attn): Physics_Attention_Structured_Mesh_3D(
            (softmax): Softmax(dim=-1)
            (dropout): Dropout(p=0.0, inplace=False)
            (in_project_x): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (in_project_fx): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (in_project_slice): Linear(in_features=8, out_features=32, bias=True)
            (to_q): Linear(in_features=8, out_features=8, bias=False)
            (to_k): Linear(in_features=8, out_features=8, bias=False)
            (to_v): Linear(in_features=8, out_features=8, bias=False)
            (to_out): Sequential(
                (0): Linear(in_features=64, out_features=64, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (ln_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
            (linear_pre): Sequential(
                (0): Linear(in_features=64, out_features=128, bias=True)
                (1): GELU(approximate='none')
            )
            (linear_post): Linear(in_features=128, out_features=64, bias=True)
            (linears): ModuleList()
            )
        )
    """

    # ---------- 内部辅助函数 ----------
    def _get_by_path(root, path: str):
        for key in path.split("."):
            root = root[int(key)] if key.isdigit(
            ) else getattr(root, key)
        return root

    def _set_by_path(root, path: str, value):
        keys = path.split(".")
        for key in keys[:-1]:
            root = root[int(key)] if key.isdigit(
            ) else getattr(root, key)
        last = keys[-1]
        if last.isdigit():
            root[int(last)] = value
        else:
            setattr(root, last, value)

    # ---------- 开始替换 ----------
    base = module.module if ddp_flag else module
    stash = []  # 记录 (path, original_submodule)

    for path in replace_layers_list:
        orig = _get_by_path(base, path)
        stash.append((path, orig))
        _set_by_path(base, path, part(checkpoint, orig))

    try:
        yield  # 进入 with 块
    finally:
        # ---------- 恢复 ----------
        for path, orig in stash:
            _set_by_path(base, path, orig)
