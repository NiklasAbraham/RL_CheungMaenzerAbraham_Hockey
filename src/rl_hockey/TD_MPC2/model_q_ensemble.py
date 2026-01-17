# Ensemble of Q-functions for value estimation.

import torch
import torch.nn as nn

from rl_hockey.TD_MPC2.model_q_function import QFunction


class QEnsemble(nn.Module):
    """
    Ensemble of Q-functions for value estimation.
    """

    def __init__(
        self, num_q=5, latent_dim=512, action_dim=8, hidden_dim=[256, 256, 256]
    ):
        super().__init__()

        self.q_functions = nn.ModuleList(
            [QFunction(latent_dim, action_dim, hidden_dim) for _ in range(num_q)]
        )

    def forward(self, latent, action):
        """
        Args:
            latent: (batch, latent_dim) current latent state
            action: (batch, action_dim) action
        Returns:
            q_list: list of (batch, 1) predicted Q-values from each Q-function
        """
        # Use ModuleList's built-in iteration which is optimized
        return [q_func(latent, action) for q_func in self.q_functions]

    def min(self, latent, action):
        """
        Args:
            latent: (batch, latent_dim) current latent state
            action: (batch, action_dim) action
        Returns:
            q: (batch,) minimum predicted Q-value
        """
        # forward() returns list of (batch, 1) tensors
        q_list = self.forward(latent, action)
        # Concatenate along dim=1 to get (batch, num_q)
        q_values = torch.cat(q_list, dim=1)  # (batch, num_q)
        # CRITICAL: torch.min()[0] returns a view - must clone to prevent inplace errors
        # .contiguous() alone doesn't create a new tensor if already contiguous
        min_val = torch.min(q_values, dim=1, keepdim=True)[0]
        return min_val.clone()  # (batch, 1) - new tensor with independent memory

    def min_subsample(self, latent, action, k=2):
        """
        Args:
            latent: (batch, latent_dim)
            action: (batch, action_dim)
            k: number of Q-heads to sample
        Returns:
            q: (batch, 1) min value among sampled heads
        """
        num_q = len(self.q_functions)
        if k > num_q:
            k = num_q
        idx = torch.randperm(num_q, device=latent.device)[:k]
        q_values = [self.q_functions[i](latent, action) for i in idx]
        q_values = torch.cat(q_values, dim=1)
        # CRITICAL: torch.min()[0] returns a view - must clone to prevent inplace errors
        min_val = torch.min(q_values, dim=1, keepdim=True)[0]
        return min_val.clone()
