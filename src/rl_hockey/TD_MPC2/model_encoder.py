import torch.nn as nn

from rl_hockey.TD_MPC2.util import SimNorm


class Encoder(nn.Module):
    """Encodes states to latent states."""

    def __init__(
        self,
        state_dim=18,
        latent_dim=512,
        hidden_dim=[256, 256, 256],
        simnorm_temperature=1.0,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim[0]))
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Mish())

        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(nn.Mish())

        layers.append(nn.Linear(hidden_dim[-1], latent_dim))
        layers.append(
            SimNorm(
                latent_dim,
                simplex_dim=min(8, latent_dim),
                temperature=simnorm_temperature,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass through encoder."""
        return self.net(state)
