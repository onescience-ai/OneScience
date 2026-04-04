import torch
import torch.nn as nn
import torch.nn.functional as F


class FourCastNetAFNO2D(nn.Module):
    """
        FourCastNet 的自适应傅里叶神经算子（AFNO）2D 混合模块。

        将输入通过 2D FFT 变换到频域，在频域内对各通道块执行稀疏化的复数 MLP 混合，
        再通过逆 FFT 还原到空间域。相比自注意力机制，AFNO 以 O(N log N) 的复杂度
        实现全局感受野的 token 混合，是 FourCastNet 替代 Transformer 注意力层的核心
        模块。参数 hard_thresholding_fraction 控制保留的频率模式比例，
        sparsity_threshold 通过软阈值化进一步稀疏化频域表示。

        Args:
            hidden_size (int, optional): 输入 token 的通道数，必须能被 num_blocks
                整除，默认为 768。
            num_blocks (int, optional): 通道分块数，每块独立在频域内做 MLP 混合，
                block_size = hidden_size // num_blocks，默认为 8。
            sparsity_threshold (float, optional): 软阈值化（soft-shrink）的阈值，
                用于在频域中稀疏化低幅度分量，默认为 0.01。
            hard_thresholding_fraction (float, optional): 保留的频率模式比例，
                仅处理低频的 kept_modes = int(H/2+1 * fraction) 个模式，
                取值范围 (0, 1]，默认为 1（保留全部模式）。
            hidden_size_factor (int, optional): 频域 MLP 中间层相对于 block_size
                的扩展倍数，默认为 1（不扩展）。

        形状:
            - 输入 x: `(Batch, Height, Width, Channels)`，其中 `Channels = hidden_size`
            - 输出:   `(Batch, Height, Width, Channels)`，形状与输入完全一致（含残差连接）

        补充说明：
            - 这里的 `Height` 与 `Width` 指的是 patch token 网格尺寸
            - 不是原始气象场分辨率，而是 `img_size // patch_size` 之后的网格尺寸

        Examples:
            >>> # 典型 FourCastNet 配置
            >>> # patch token 网格尺寸 90×180，hidden_size=768，分为8块（block_size=96）
            >>> # total_modes = 90 // 2 + 1 = 46
            >>> # kept_modes  = int(46 * 1.0) = 46（保留全部频率模式）
            >>> afno = FourCastNetAFNO2D(
            ...     hidden_size=768,
            ...     num_blocks=8,
            ...     sparsity_threshold=0.01,
            ...     hard_thresholding_fraction=1,
            ...     hidden_size_factor=1,
            ... )
            >>> x = torch.randn(2, 90, 180, 768)
            >>> out = afno(x)
            >>> out.shape
            torch.Size([2, 90, 180, 768])
    """

    def __init__(
        self,
        hidden_size=768,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
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
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        Batch, Height, Width, Channels = x.shape

        if Channels != self.hidden_size:
            raise ValueError(
                f"Expected input channels {self.hidden_size}, but received {Channels}"
            )

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(Batch, Height, Width // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                Batch,
                Height,
                Width // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                Batch,
                Height,
                Width // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = Height // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].real,
                self.w1[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].imag,
                self.w1[1],
            )
            + \
            self.b1[0]
        )

        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].imag,
                self.w1[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                x[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes].real,
                self.w1[1],
            )
            + \
            self.b1[1]
        )

        o2_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[1],
            )
            + \
            self.b2[0]
        )

        o2_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],
                self.w2[1],
            )
            + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(Batch, Height, Width // 2 + 1, Channels)
        x = torch.fft.irfft2(x, s=(Height, Width), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias
