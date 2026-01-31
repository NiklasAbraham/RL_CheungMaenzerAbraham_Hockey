"""Neural network for decoy policy that predicts actions from observations."""

import torch
import torch.nn as nn


class DecoyPolicyNetwork(nn.Module):
    """Neural network for decoy policy that predicts actions from observations."""

    def __init__(
        self,
        obs_dim=18,
        action_dim=4,
        hidden_layers=[256, 256],
        log_std_min=-10.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        layers.append(nn.Linear(obs_dim, hidden_layers[0]))
        layers.append(nn.LayerNorm(hidden_layers[0]))
        layers.append(nn.Mish())

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.LayerNorm(hidden_layers[i]))
            layers.append(nn.Mish())

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_layers[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_layers[-1], action_dim)

    def _distribution(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std_raw = self.log_std_head(h)
        log_std_dif = self.log_std_max - self.log_std_min
        log_std = self.log_std_min + 0.5 * log_std_dif * (torch.tanh(log_std_raw) + 1)
        std = torch.exp(log_std)
        return mean, std, log_std

    def mean_action(self, obs):
        """Returns the mean action (deterministic policy)."""
        mean, _, _ = self._distribution(obs)
        return torch.tanh(mean)

    def sample(self, obs):
        """Sample action from policy with log probability."""
        mean, std, log_std = self._distribution(obs)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps

        gaussian_log_prob = -0.5 * eps.pow(2) - log_std - 0.9189385175704956
        gaussian_log_prob = gaussian_log_prob.sum(dim=-1, keepdim=True)

        action = torch.tanh(pre_tanh)
        squash_correction = torch.log(torch.relu(1 - action.pow(2)) + 1e-6).sum(
            dim=-1, keepdim=True
        )
        log_prob = gaussian_log_prob - squash_correction

        return action, log_prob

    def forward(self, obs):
        """Forward pass returns mean action."""
        return self.mean_action(obs)
