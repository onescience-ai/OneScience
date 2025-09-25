import torch
import torch.nn as nn

from onescience.utils.layers.evolution.evolution_network import Evolution_Network
from onescience.utils.layers.generation.generative_network import (
    Generative_Decoder,
    Generative_Encoder,
)
from onescience.utils.layers.generation.noise_projector import Noise_Projector
from onescience.utils.layers.utils import make_grid, warp


class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length

        self.evo_net = Evolution_Network(
            self.configs.input_length, self.pred_length, base_c=32
        )
        self.gen_enc = Generative_Encoder(
            self.configs.total_length, base_c=self.configs.ngf
        )
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)

        sample_tensor = torch.zeros(
            1, 1, self.configs.img_height, self.configs.img_width
        )
        self.grid = make_grid(sample_tensor)

    def forward(self, all_frames):
        all_frames = all_frames[:, :, :, :, :1]

        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        input_frames = frames[:, : self.configs.input_length]
        input_frames = input_frames.reshape(
            batch, self.configs.input_length, height, width
        )

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)

        series = []
        bilis = []
        last_frames = all_frames[
            :, (self.configs.input_length - 1) : self.configs.input_length, :, :, 0
        ]

        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):

            last_frame_bili = warp(
                last_frames,
                motion_[:, i],
                grid.to(self.configs.device),
                mode="bilinear",
                padding_mode="border",
            )
            bilis.append(last_frame_bili)

            last_frames = warp(
                last_frames,
                motion_[:, i],
                grid.to(self.configs.device),
                mode="nearest",
                padding_mode="border",
            )
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)

        evo_result = torch.cat(series, dim=1)
        evo_motion = torch.cat(bilis, dim=1)

        evo_result = evo_result / 128

        return evo_result, evo_motion, motion_
