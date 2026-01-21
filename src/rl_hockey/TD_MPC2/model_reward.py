import torch
import torch.nn as nn


class Reward(nn.Module):
    """
    Predicts reward given latent state and action.
    """

    def __init__(self, latent_dim=512, action_dim=8, hidden_dim=[256, 256, 256], num_bins=101, vmin=-10.0, vmax=10.0):
        super().__init__()

        layers = []
        layers.append(nn.Linear(latent_dim + action_dim, hidden_dim[0]))
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Mish())

        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(nn.Mish())

        layers.append(nn.Linear(hidden_dim[-1], num_bins))

        self.net = nn.Sequential(*layers)
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, latent, action):
        """Forward pass through reward model."""
        x = torch.cat([latent, action], dim=-1)

        return self.net(x)
