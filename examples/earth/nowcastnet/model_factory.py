import torch

from onescience.models.nowcastnet import (
    discriminator2,
    evolutionnet,
    generator,
    nowcastnet,
)


class Model(object):
    def __init__(self, configs, mode):
        self.configs = configs
        self.mode = mode
        networks_map = {
            "NowcastNet": nowcastnet.Net,
            "EvolutionNet": evolutionnet.Net,
            "Generator": generator.Net,
            "Discriminator": discriminator2.Net,
        }
        self.data_frame = []

        if mode in networks_map:
            Network = networks_map[mode]
            self.network = Network(
                configs).to(configs.device)
        else:
            raise ValueError(
                "Name of network unknown %s" % mode)

    def test_load(self):

        for name, param in self.network.named_parameters():
            print(name, param.requires_grad)

        stats = torch.load(self.configs.pretrained_model)
        new_state_dict = self.network.state_dict()
        new_state_dict.update(stats)
        self.network.load_state_dict(new_state_dict)

        for name, param in self.network.named_parameters():
            print(name, param.requires_grad)

    def test(self, frames):
        frames_tensor = torch.FloatTensor(
            frames).to(self.configs.device)
        self.network.eval()
        with torch.no_grad():
            next_frames = self.network(frames_tensor)
        return next_frames.detach().cpu().numpy()

    def train(self, frames):
        if not isinstance(frames, torch.Tensor):
            frames_tensor = torch.FloatTensor(
                frames).to(self.configs.device)
        else:
            frames_tensor = frames.to(self.configs.device)
        self.network.train()
        next_frames = self.network(frames_tensor)
        return next_frames

    def valid(self, frames):
        if not isinstance(frames, torch.Tensor):
            frames_tensor = torch.FloatTensor(
                frames).to(self.configs.device)
        else:
            frames_tensor = frames.to(self.configs.device)
        self.network.eval()
        with torch.no_grad():
            next_frames = self.network(frames_tensor)
        return next_frames
