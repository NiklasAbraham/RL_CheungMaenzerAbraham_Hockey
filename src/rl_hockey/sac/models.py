import torch
import torch.nn as nn
from torch.distributions import Normal


# TODO: use BatchRenorm instead
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dims, activation, use_batchnorm=False):
        super().__init__()

        layers = []
        dim = input_dim
        for latent_dim in latent_dims:
            layers.append(nn.Linear(dim, latent_dim))
            layers.append(activation())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(latent_dim))
            dim = latent_dim
        layers.append(nn.Linear(dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dims=[256, 256], activation=nn.ReLU):
        super().__init__()

        self.net = MLP(state_dim + action_dim, 1, latent_dims, activation)

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.net(input)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dims=[256, 256], activation=nn.ReLU):
        super().__init__()

        self.net = MLP(state_dim, 2 * action_dim, latent_dims, activation)

    def forward(self, state):
        output = self.net(state)
        mean, log_std = torch.chunk(output, 2, dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state, noise=None, calc_log_prob=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if noise is None:
            noise = torch.randn_like(mean)

        action = mean + std*noise
        action_squashed = torch.tanh(action)

        log_prob = None
        if calc_log_prob:
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action)
            log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6)    # correct for tanh squashing
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action_squashed, log_prob
