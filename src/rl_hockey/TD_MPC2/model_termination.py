import torch
import torch.nn as nn


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

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.mlp(x)
