import torch
import torch.nn as nn


class QFunction(nn.Module):
    """
    Q-function for value estimation.
    """

    def __init__(
        self, latent_dim=512, action_dim=8, hidden_dim=[256, 256, 256], num_bins=101
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(latent_dim + action_dim, hidden_dim[0]))
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Dropout(p=0.01))
        layers.append(nn.Mish())

        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(nn.Mish())

        layers.append(nn.Linear(hidden_dim[-1], num_bins))

        self.net = nn.Sequential(*layers)
        self.num_bins = num_bins

    def forward(self, latent, action):
        """Forward pass through Q-function."""
        x = torch.cat([latent, action], dim=-1)
        q_logits = self.net(x)
        return q_logits
