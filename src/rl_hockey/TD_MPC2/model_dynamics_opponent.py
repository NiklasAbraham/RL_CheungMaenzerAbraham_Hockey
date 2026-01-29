import torch
import torch.nn as nn

from rl_hockey.TD_MPC2.util import SimNorm


class DynamicsOpponent(nn.Module):
    """Predicts next latent state given current latent state and action."""

    def __init__(
        self,
        latent_dim=512,
        action_dim=8,
        action_opponent_dim=8,
        hidden_dim=[256, 256, 256],
        simnorm_temperature=1.0,
    ):
        super().__init__()

        layers = []

        layers.append(
            nn.Linear(latent_dim + action_dim + action_opponent_dim, hidden_dim[0])
        )
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Mish())

        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(nn.Mish())

        layers.append(nn.Linear(hidden_dim[-1], latent_dim))
        layers.append(nn.LayerNorm(latent_dim))

        self.net = nn.Sequential(*layers)
        self.simnorm = SimNorm(
            latent_dim, simplex_dim=min(8, latent_dim), temperature=simnorm_temperature
        )

    def forward(self, latent, action, action_opponent):
        """Forward pass through dynamics model."""
        x = torch.cat([latent, action, action_opponent], dim=-1)
        z_after_linear = self.net(x)
        latent_next = self.simnorm(z_after_linear)

        return latent_next
