# reference: https://github.com/NVlabs/AFNO-transformer
from functools import partial

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    # 在频域内进行稀疏注意力机制。
    # 通过傅里叶变换和逆傅里叶变换实现输入张量在频域内的变换。
    # 使用全连接层进行非线性变换和稀疏处理。

    # hiddensize是嵌入维度768
    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):
        super().__init__()
        # 后面要将嵌入值按照通道分块，所以必须要能被整除
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks

        # 每一块的大小是768/8=96
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        # 输入为batch, 14, 14, 768
        # 记录输入，后面用来做残差计算
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        # 进行FFT，输出为batch, 14, 8, 768，因为频谱对称所以FFT只计算一半的频谱
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        # 重组数据，拆分为batch, 14, 8, 8，96
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        # 计算实部
        o1_real = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)

        # 计算虚部
        o1_imag = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # 爱因斯坦求和，可以根据第一个参数灵活的计算
        # o1 real的维度为 B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor
        # 实际是             B 14       8                 8,                           96*1
        # 选择要保留的频率成分，total_modes - kept_modes 到 total_modes + kept_modes 范围内的频率
        # 实际上total_modes == kept_modes，因此保留的就是全部
        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(
                torch.einsum(
                    "...bi,bio->...bo",
                    # x 的形状，... 表示批次维度，b 表示频率维度，i 表示输入通道数。
                    # 权重矩阵 w 的形状，b 表示频率维度，i 表示输入通道数，o 表示输出通道数。
                    # 输出张量的形状，... 表示批次维度，b 表示频率维度，o 表示输出通道数。
                    # 输入维度：[B, 14, 8, 8, 96]
                    # 权重维度：[8, 96, 96]
                    # 输出维度：[B, 14, 8, 8, 96]
                    x[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ].real,
                    self.w1[0],
                )
                - torch.einsum(
                    "...bi,bio->...bo",
                    x[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ].imag,
                    self.w1[1],
                )
                + self.b1[0]
            )
        )

        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(
                torch.einsum(
                    "...bi,bio->...bo",
                    x[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ].imag,
                    self.w1[0],
                )
                + torch.einsum(
                    "...bi,bio->...bo",
                    x[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ].real,
                    self.w1[1],
                )
                + self.b1[1]
            )
        )

        o2_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        # 将实部和虚部堆叠在一起
        x = torch.stack([o2_real, o2_imag], dim=-1)

        # 稀疏处理
        x = F.softshrink(x, lambd=self.sparsity_threshold)

        # 从实部和虚部转换为复数表示
        x = torch.view_as_complex(x)

        # 重塑回四维
        x = x.reshape(B, H, W // 2 + 1, C)

        # 进行逆傅里叶变换，结果是一个实数域张量
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
        self,
        dim,  # 嵌入的维度，目前为768
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(
            dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x):
        # 输入为batch, 14, 14, 768

        # 保留初始值用于残差计算
        residual = x

        x = self.norm1(x)
        # 进行layer norm，输出batch, 14, 14, 768
        x = self.filter(x)
        # 进行FFT， 输出batch, 14, 14, 768

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNONet(nn.Module):
    def __init__(
        self,
        params,
        patch_size=(16, 16),
        in_chans=2,
        out_chans=2,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.params = params
        self.depth = params.depth
        self.img_size = (params.image_width, params.image_height)
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = len(params.in_channels)
        self.out_chans = len(params.out_channels)
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(self.depth)
            ]
        )
        # self.norm = norm_layer(embed_dim)

        # 输入为768， 输出为目标通道*patch长*patch宽，用于后续重组
        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]
        # torch.Size([5, 72, 160, 360])
        x = self.patch_embed(x)
        # torch.Size([5, 3600, 768])

        x = x + self.pos_embed
        # torch.Size([5, 3600, 768])
        x = self.pos_drop(x)
        # torch.Size([5, 3600, 768])

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        # torch.Size([5, 40, 90, 768])
        for blk in self.blocks:
            x = blk(x)
        # torch.Size([5, 40, 90, 768])
        return x

    def forward(self, x):
        # print('step 0:', x.shape)
        x = self.forward_features(x)  # 划分patch以及添加位置编码等
        # print('step 1:', x.shape)
        # 输出batch, 14, 14, 768

        x = self.head(x)
        # print('step 2:', x.shape)
        #  输出为batch, 14, 14，目标通道*patch长*patch宽，用于后续重组

        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        # print('step 3:', x.shape)
        # 从batch, 14, 14, 目标通道*patch长*patch宽 变为 batch, 目标通道，14*16=224, 14*16=224
        return x


class PatchEmbed(nn.Module):
    # 将一张图划分为patch
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 用卷积来编码
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # print('==================================== ')
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # 输入为batch, 3, 224, 224，patch大小为16，16
        # 首先用卷积编码后，输出batch, 768, 14, 14，其中14=224/16
        # 随后将14，14的patch展平为batch, 768, 196
        # 最后将2，3维度交换，变为batch, 196, 768，与transformer结合，代表有196个token，嵌入为768向量
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
