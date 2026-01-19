# Ensemble of Q-functions.

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
        """
        Args:
            latent: (batch, latent_dim) current latent state
            action: (batch, action_dim) action
        Returns:
            q_list: list of (batch, num_bins) predicted Q-value logits from each Q-function
        """
        q_list = [q_func(latent, action) for q_func in self.q_functions]
        return q_list

    def min(self, latent, action):
        """
        Args:
            latent: (batch, latent_dim) current latent state
            action: (batch, action_dim) action
        Returns:
            q: (batch, 1) minimum predicted Q-value (converted from logits)
        """
        q_list = self.forward(latent, action)
        q_values = torch.cat(
            [
                two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax)
                for q_logits in q_list
            ],
            dim=1,
        )
        min_val = q_values.min(dim=1, keepdim=True)[0]
        return min_val

    def min_subsample(self, latent, action, k=2):
        """
        Args:
            latent: (batch, latent_dim)
            action: (batch, action_dim)
            k: number of Q-heads to sample
        Returns:
            q: (batch, 1) min value among sampled heads (converted from logits)
        """
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
        """
        Return average Q-value from k randomly sampled Q-heads.

        Args:
            latent: (batch, latent_dim)
            action: (batch, action_dim)
            k: number of Q-heads to sample
        Returns:
            q: (batch, 1) average value among sampled heads (converted from logits)
        """
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

    def avg_subsample_detached(self, latent, action, k=2):
        """
        Return average Q-value from k randomly sampled Q-heads for policy update.

        Args:
            latent: (batch, latent_dim)
            action: (batch, action_dim)
            k: number of Q-heads to sample
        Returns:
            q: (batch, 1) average value among sampled heads
        """
        num_q = len(self.q_functions)
        if k > num_q:
            k = num_q
        idx = torch.randperm(num_q, device=latent.device)[:k]
        idx_list = idx.cpu().tolist()
        q_logits_list = [self.q_functions[i](latent.detach(), action) for i in idx_list]
        q_values = torch.cat(
            [
                two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax)
                for q_logits in q_logits_list
            ],
            dim=1,
        )
        avg_val = q_values.mean(dim=1, keepdim=True)
        return avg_val
