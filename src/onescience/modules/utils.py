from torch import nn
import torch
import math

def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """改编自 timm master
    按样本丢弃路径（Drop paths / 随机深度 Stochastic Depth）（当应用于残差块的主路径时）
    这与我为 EfficientNet 等网络实现的 DropConnect 功能相同，但原来的命名容易引起误解，因为 “Drop Connect” 
    在另一篇论文中指的是不同形式的 Dropout，参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956，
    选择将层和参数名称改为 Drop Path，而不是沿用 DropConnect 作为层名，同时将参数命名为 survival rate（存活率）。

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """摘自 timm 仓库
    按样本丢弃路径（Drop paths / 随机深度 Stochastic Depth）当应用于残差块的主路径时）
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


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

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_earth_position_index(window_size, ndim=3):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    构建位置索引（Position Index）以复用位置偏置的对称参数
    实现参考: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    参数:
        window_size (tuple[int]): 窗口大小，三维为 [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): 张量维度，2 表示二维窗口，3 表示三维窗口

    返回值:
        position_index (torch.Tensor): 位置索引矩阵，形状为：[win_pl * win_lat * win_lon, win_pl * win_lat * win_lon] or [win_lat * win_lon, win_lat * win_lon]
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
    elif ndim == 2:
        win_lat, win_lon = window_size

    if ndim == 3:
        # Index in the pressure level of query matrix
        coords_zi = torch.arange(win_pl)
        # Index in the pressure level of key matrix
        coords_zj = -torch.arange(win_pl) * win_pl

    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # Change the order of the index to calculate the index in total
    if ndim == 3:
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    elif ndim == 2:
        coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    if ndim == 3:
        coords[:, :, 2] += win_lon - 1
        coords[:, :, 1] *= 2 * win_lon - 1
        coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    elif ndim == 2:
        coords[:, :, 1] += win_lon - 1
        coords[:, :, 0] *= 2 * win_lon - 1

    # Sum up the indexes in two/three dimensions
    position_index = coords.sum(-1)

    return position_index

def save_checkpoint(
    model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path
):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    torch.save(state, f"{model_path}/pangu_weather.pth")

def get_pad3d(input_resolution, window_size):
    """
    参数:
        input_resolution (tuple[int]): 输入数据的分辨率 (Pl, Lat, Lon)，分别表示气压层、纬度和经度方向的大小。
        window_size (tuple[int]): 窗口大小 (Pl, Lat, Lon)，用于计算分块或滑动窗口时的尺寸。

    返回:
        padding (tuple[int]): 填充大小 (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)，用于保证输入数据可以整除窗口大小或适配滑动窗口。
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = (
        padding_right
    ) = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def get_pad2d(input_resolution, window_size):
    """
    参数:
        input_resolution (tuple[int]): 输入数据的分辨率 (Lat, Lon)，分别表示纬度和经度方向的大小。
        window_size (tuple[int]): 窗口大小 (Lat, Lon)，用于计算分块或滑动窗口时的尺寸

    返回:
        padding (tuple[int]): 填充大小 (padding_left, padding_right, padding_top, padding_bottom)，用于保证输入数据可以整除窗口大小或适配滑动窗口。
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


def crop2d(x: torch.Tensor, resolution):
    """
    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, Lat, Lon)，其中 B 为批量大小，C 为通道数，Lat 和 Lon 分别为纬度和经度方向的尺寸。
        resolution (tuple[int]): 输入分辨率 (Lat, Lon)，对应纬度和经度方向的尺寸。
    """
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[
        :, :, padding_top : Lat - padding_bottom, padding_left : Lon - padding_right
    ]


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): 输入张量，形状为 (B, C, Pl, Lat, Lon)，其中 B 为批量大小，C 为通道数，Pl 为气压层数量，Lat 和 Lon 分别为纬度和经度方向的尺寸。
        resolution (tuple[int]): 输入分辨率 (Pl, Lat, Lon)，对应气压层、纬度和经度方向的尺寸。
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    u1 = norm_cdf((a - mean) / std)
    u2 = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [u1, u2], then translate to
    # [2u1-1, 2u2-1].
    tensor.uniform_(2 * u1 - 1, 2 * u2 - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""从 timm 主分支中剪切粘贴而来。
    该函数使用截断正态分布为输入张量填充值。
    这些值实际上取自正态分布:math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    但若超出区间 [a, b]，则会重新采样，直到落入范围内为止。
    当满足 a ≤ mean ≤ b 时，该方法的效果最佳。:math:`a \leq \text{mean} \leq b`.

    NOTE: 注意：该实现与 PyTorch 的 trunc_normal_ 类似，
    但截断区间 [a, b] 是在对 mean 和 std 应用之后才生效，
    因此参数 a、b 应根据 mean 和 std 的范围进行调整。
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def window_partition(x: torch.Tensor, window_size, ndim=3):
    """
    参数:
        x: 输入张量，形状为 (B, Pl, Lat, Lon, C) 或 (B, Lat, Lon, C)。
        window_size (tuple[int]): 窗口大小，格式为 [win_pl, win_lat, win_lon] 或 [win_lat, win_lon]。
        ndim (int): 窗口的维度，取值为 3 或 2。

    返回:
        windows: 分块后的窗口张量，形状为(B × num_lon, num_pl × num_lat, win_pl, win_lat, win_lon, C) 或(B × num_lon, num_lat, win_lat, win_lon, C)。
    """
    if ndim == 3:
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size
        x = x.view(
            B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C
        )
        windows = (
            x.permute(0, 5, 1, 3, 2, 4, 6, 7)
            .contiguous()
            .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
        )
        return windows
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size
        x = x.view(B, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = (
            x.permute(0, 3, 1, 2, 4, 5)
            .contiguous()
            .view(-1, (Lat // win_lat), win_lat, win_lon, C)
        )
        return windows


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    参数:
        windows: 输入窗口张量，形状为(B × num_lon, num_pl × num_lat, win_pl, win_lat, win_lon, C) 或(B × num_lon, num_lat, win_lat, win_lon, C)。
        window_size (tuple[int]): 窗口大小，格式为 [win_pl, win_lat, win_lon] 或 [win_lat, win_lon]。
        Pl (int): 气压层数（pressure levels）。
        Lat (int): 纬度（latitude）。
        Lon (int): 经度（longitude）。
        ndim (int): 窗口维度，取值为 3 或 2。
    返回值:
        x: 重建后的张量，形状为 (B, Pl, Lat, Lon, C) 或 (B, Lat, Lon, C)。
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(
            B,
            Lon // win_lon,
            Pl // win_pl,
            Lat // win_lat,
            win_pl,
            win_lat,
            win_lon,
            -1,
        )
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
        return x
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Lat // win_lat, win_lat, win_lon, -1)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, Lat, Lon, -1)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    沿着经度（longitude）维度，最左和最右的索引实际上是相邻的。
    如果在最左和最右位置都出现了半个窗口（half window），则它们会被直接合并为一个完整窗口。
    参数:
        input_resolution (tuple[int]): 输入的分辨率，格式为 [pressure levels, latitude, longitude] 或 [latitude, longitude]。
        window_size (tuple[int]): 窗口大小，格式为 [pressure levels, latitude, longitude] 或 [latitude, longitude]。
        shift_size (tuple[int]): 用于SW-MSA的窗口平移大小，格式为 [pressure levels, latitude, longitude] 或 [latitude, longitude]。
        ndim (int): 窗口的维度，取值为 3 或 2

    返回:
        attn_mask: 注意力掩码张量，形状为(n_lon, n_pl * n_lat, win_pl * win_lat * win_lon, win_pl * win_lat * win_lon)或(n_lon, n_lat, win_lat * win_lon, win_lat * win_lon)。
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))

    if ndim == 3:
        pl_slices = (
            slice(0, -win_pl),
            slice(-win_pl, -shift_pl),
            slice(-shift_pl, None),
        )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )

    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]

    mask_windows = window_partition(
        img_mask, window_size, ndim=ndim
    )  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1 or n_lon, n_lat, win_lat, win_lon, 1
    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon
    mask_windows = mask_windows.view(
        mask_windows.shape[0], mask_windows.shape[1], win_total
    )
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask
