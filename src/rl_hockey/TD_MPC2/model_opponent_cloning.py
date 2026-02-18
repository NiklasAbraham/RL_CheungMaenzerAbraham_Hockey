"""Opponent cloning network (policy in latent space)."""

import torch
import torch.nn as nn


class OpponentCloning(nn.Module):
    """
    Opponent cloning network.
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
        layers.append(nn.Linear(latent_dim, hidden_dim[0]))
        layers.append(nn.LayerNorm(hidden_dim[0]))
        layers.append(nn.Mish())

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
        log_std_raw = self.log_std_head(h)
        log_std_dif = self.log_std_max - self.log_std_min
        log_std = self.log_std_min + 0.5 * log_std_dif * (torch.tanh(log_std_raw) + 1)
        std = torch.exp(log_std)
        return mean, std, log_std

    def mean_action(self, latent):
        mean, _, _ = self._distribution(latent)
        return torch.tanh(mean)

    def sample(self, latent):
        """Sample action from policy."""
        mean, std, log_std = self._distribution(latent)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps

        gaussian_log_prob = -0.5 * eps.pow(2) - log_std - 0.9189385175704956
        gaussian_log_prob = gaussian_log_prob.sum(dim=-1, keepdim=True)
        scaled_log_prob = gaussian_log_prob * mean.shape[-1]

        mean_action = torch.tanh(mean)
        action = torch.tanh(pre_tanh)
        squash_correction = torch.log(torch.relu(1 - action.pow(2)) + 1e-6).sum(
            dim=-1, keepdim=True
        )
        log_prob = gaussian_log_prob - squash_correction
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        scaled_entropy = -log_prob * entropy_scale

        return action, log_prob, mean_action, scaled_entropy

    def forward(self, latent):
        return self.mean_action(latent)
