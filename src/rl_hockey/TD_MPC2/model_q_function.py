# Q-function for value estimation.

import torch
import torch.nn as nn


class QFunction(nn.Module):
    """
    Q-function for value estimation.
    """

    def __init__(self, latent_dim=512, action_dim=8, hidden_dim=[256, 256, 256]):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(latent_dim + action_dim, hidden_dim[0]))
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Mish())

        # Hidden layers
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(nn.Mish())

        # Output layer (scalar Q-value)
        layers.append(nn.Linear(hidden_dim[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, latent, action):
        """
        Args:
            latent: (batch, latent_dim) current latent state
            action: (batch, action_dim) action
        Returns:
            q: (batch, 1) predicted Q-value
        """
        x = torch.cat([latent, action], dim=-1)
        return self.net(x)
