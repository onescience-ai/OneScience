from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from .base_model import AutoCfdModel
from onescience.modules import OneEncoder, OneDecoder, OneHead

class UNet(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        loss_fn: nn.Module,
        n_case_params: int,
        insert_case_params_at: str = "hidden",
        bilinear: bool = False,
        dim: int = 8,
    ):
        assert insert_case_params_at in ["hidden", "input"]
        super().__init__(loss_fn)
        
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.insert_case_params_at = insert_case_params_at
        self.dim = dim

        # 计算 Encoder 输入通道
        encoder_in_chan = in_chan + 1 # + Mask
        if insert_case_params_at == "input":
            encoder_in_chan += n_case_params

        # 1. Encoder
        self.encoder = OneEncoder(
            style="UNetEncoder2D",
            in_channels=encoder_in_chan,
            base_channels=dim,
            num_stages=4,
            bilinear=bilinear,
            normtype="bn"
        )

        # 2. Hidden Injection
        self.case_params_fc = None
        if insert_case_params_at == "hidden":
            bottleneck_dim = dim * 16
            self.case_params_fc = nn.Linear(n_case_params, bottleneck_dim)

        # 3. Decoder
        self.decoder = OneDecoder(
            style="UNetDecoder2D",
            base_channels=dim,
            num_stages=4,
            bilinear=bilinear,
            normtype="bn"
        )

        # 4. Head
        self.head = OneHead(
            style="UNetHead2D",
            in_channels=dim,
            out_channels=out_chan
        )

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ):
        batch_size, n_chan, height, width = inputs.shape
        residual = inputs[:, : self.out_chan]

        # 构造 Mask
        if mask is None:
            mask = torch.ones((batch_size, 1, height, width)).to(inputs.device)
        else:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) # (B, H, W) -> (B, 1, H, W)
        
        # 拼接 Mask
        x_in = torch.cat([inputs, mask], dim=1)

        # 拼接 Case Params
        if self.insert_case_params_at == "input":
            cp_spatial = case_params.view(batch_size, self.n_case_params, 1, 1)
            cp_spatial = cp_spatial.expand(-1, -1, height, width)
            x_in = torch.cat([x_in, cp_spatial], dim=1)

        # Encoder
        features = self.encoder(x_in)
        
        # Hidden 注入
        if self.insert_case_params_at == "hidden":
            bottleneck = features[-1]
            conds = self.case_params_fc(case_params)
            conds = conds.view(batch_size, -1, 1, 1)
            features[-1] = bottleneck + conds

        # Decoder
        decoded = self.decoder(features)

        # Head
        preds = self.head(decoded)
        
        # Residual & Mask
        preds = preds + residual
        preds = preds * mask

        if label is not None:
            label = label * mask
            loss = self.loss_fn(labels=label, preds=preds)
            return {"preds": preds, "loss": loss}
        
        return {"preds": preds}

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        preds = []
        
        # 处理单样本输入 (增加 Batch 维)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            case_params = case_params.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

        # 确保 Mask 是 (B, 1, H, W) 以匹配 forward 逻辑
        if mask.dim() == 3:
             mask = mask.unsqueeze(1) 

        cur_frame = inputs
        for _ in range(steps):
            out_dict = self.forward(cur_frame, case_params=case_params, mask=mask)
            cur_frame = out_dict["preds"]
            preds.append(cur_frame)
            
        return preds

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = self.forward(inputs, case_params=case_params, mask=mask)
        return outputs["preds"]