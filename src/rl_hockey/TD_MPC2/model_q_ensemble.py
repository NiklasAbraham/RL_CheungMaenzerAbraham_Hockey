import torch
import torch.nn as nn

from rl_hockey.TD_MPC2.model_q_function import QFunction
from rl_hockey.TD_MPC2.util import two_hot_inv


class QEnsemble(nn.Module):
    """
    Ensemble of Q-functions for value estimation.
    """

    def __init__(
        self,
        num_q=5,
        latent_dim=512,
        action_dim=8,
        hidden_dim=[256, 256, 256],
        num_bins=101,
        vmin=-100.0,
        vmax=100.0,
    ):
        super().__init__()

        self.q_functions = nn.ModuleList(
            [
                QFunction(latent_dim, action_dim, hidden_dim, num_bins)
                for _ in range(num_q)
            ]
        )
        self.num_q = num_q
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, latent, action):
        """Forward pass through Q-ensemble."""
        q_list = [q_func(latent, action) for q_func in self.q_functions]
        return q_list

    def min(self, latent, action):
        """Returns minimum Q-value from a subsampled ensemble."""
        return self.min_subsample(latent, action, k=2)

    def min_subsample(self, latent, action, k=2):
        """Returns minimum Q-value from k sampled heads."""
        num_q = len(self.q_functions)
        if k > num_q:
            k = num_q
        idx = torch.randperm(num_q, device=latent.device)[:k]
        idx_list = idx.cpu().tolist()
        q_logits_list = [self.q_functions[i](latent, action) for i in idx_list]
        q_values = torch.cat(
            [
                two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax)
                for q_logits in q_logits_list
            ],
            dim=1,
        )
        min_val = torch.min(q_values, dim=1, keepdim=True)[0]
        return min_val

    def avg_subsample(self, latent, action, k=2):
        """Returns average Q-value from k sampled heads."""
        num_q = len(self.q_functions)
        if k > num_q:
            k = num_q
        idx = torch.randperm(num_q, device=latent.device)[:k]
        idx_list = idx.cpu().tolist()
        q_logits_list = [self.q_functions[i](latent, action) for i in idx_list]
        q_values = torch.cat(
            [
                two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax)
                for q_logits in q_logits_list
            ],
            dim=1,
        )
        avg_val = q_values.mean(dim=1, keepdim=True)
        return avg_val

    def avg(self, latent, action):
        """Returns average Q-value from a subsampled ensemble."""
        return self.avg_subsample(latent, action, k=2)
