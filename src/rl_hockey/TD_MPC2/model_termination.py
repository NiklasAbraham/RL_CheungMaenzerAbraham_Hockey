import torch.nn as nn
import torch

from rl_hockey.TD_MPC2.model_init import weight_init


class Termination(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1),
        )
        self.apply(weight_init)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.mlp(x)
