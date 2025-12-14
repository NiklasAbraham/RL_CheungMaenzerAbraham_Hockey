import torch
import torch.nn as nn
from torch.distributions import Normal


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, activation=nn.ReLU):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.net(input)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, activation=nn.ReLU):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, 2 * action_dim),
        )

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
