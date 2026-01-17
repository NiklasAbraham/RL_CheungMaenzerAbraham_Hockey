import torch
import torch.nn as nn

from rl_hockey.TD_MPC2.util import SimNorm


class DynamicsSimple(nn.Module):
    """
    Predicts next latent state given current latent state and action.

    Architecture from TD-MPC2 paper:
    - Input: current latent state and action
    - Output: next latent state
    - Residual connection: next latent state = current latent state + predicted change
    """

    def __init__(
        self,
        latent_dim=512,
        action_dim=8,
        hidden_dim=[256, 256, 256],
        simnorm_temperature=1.0,
    ):
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

        # Output layer
        layers.append(nn.Linear(hidden_dim[-1], latent_dim))

        self.net = nn.Sequential(*layers)
        self.simnorm = SimNorm(
            latent_dim, simplex_dim=min(8, latent_dim), temperature=simnorm_temperature
        )

    def forward(self, latent, action):
        """

        Architecture from TD-MPC2 paper:
        - Input: current latent state and action
        - Output: next latent state
        - Residual connection: next latent state = current latent state + predicted change

        Args:
            latent: (batch, latent_dim) current latent state
            action: (batch, action_dim) action
        Returns:
            latent_next: (batch, latent_dim) next latent state
        """
        x = torch.cat([latent, action], dim=-1)

        delta_z = self.net(x)

        latent_next = latent + delta_z
        return self.simnorm(latent_next)
