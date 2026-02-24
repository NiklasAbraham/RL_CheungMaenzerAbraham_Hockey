
import torch
from torch.distributions import Normal
from pink import ColoredNoiseProcess


class NormalNoise:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.dist = Normal(torch.zeros(action_dim), torch.ones(action_dim))

    def sample(self):
        return self.dist.sample()
    

class PinkNoise:
    def __init__(self, action_dim, max_sequence_length):
        self.dist = ColoredNoiseProcess(1, (action_dim, max_sequence_length))

    def sample(self):
        noise = self.dist.sample()
        return torch.from_numpy(noise)

    def reset(self):
        self.dist.reset()


class RedNoise:
    def __init__(self, action_dim, max_sequence_length):
        self.dist = ColoredNoiseProcess(2, (action_dim, max_sequence_length))

    def sample(self):
        noise = self.dist.sample()
        return torch.from_numpy(noise)

    def reset(self):
        self.dist.reset()
