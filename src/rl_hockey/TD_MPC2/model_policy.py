# Policy network for action initialization in MPC.

import torch
import torch.nn as nn


class Policy(nn.Module):
    """
    Stochastic policy prior for action initialization in MPC.
    """

    def __init__(
        self,
        latent_dim=512,
        action_dim=8,
        hidden_dim=[256, 256, 256],
        log_std_min=-10.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []

        # Input layer
        layers.append(nn.Linear(latent_dim, hidden_dim[0]))
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Mish())

        # Hidden layers
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(nn.Mish())

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dim[-1], action_dim)

    def _distribution(self, latent):
        h = self.net(latent)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std, log_std

    def mean_action(self, latent):
        mean, _, _ = self._distribution(latent)
        return torch.tanh(mean)

    def sample(self, latent):
        mean, std, log_std = self._distribution(latent)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)

        # Tanh-squashed Gaussian log-prob
        log_prob = -0.5 * (
            ((pre_tanh - mean) / (std + 1e-6)) ** 2
            + 2 * log_std
            + torch.log(torch.tensor(2.0 * torch.pi, device=pre_tanh.device))
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Use non-inplace operation to avoid gradient computation errors
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )
        return action, log_prob, torch.tanh(mean)

    def forward(self, latent):
        return self.mean_action(latent)
