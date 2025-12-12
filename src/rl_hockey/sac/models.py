import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.net(input)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256):
        super().__init__()

    def forward(self, state):
        raise NotImplementedError()
