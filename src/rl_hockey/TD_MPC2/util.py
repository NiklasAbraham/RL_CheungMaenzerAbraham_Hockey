# utility functions for the TD-MPC2 implementation
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """
    Simplicial Normalization layer from TD-MPC2 paper.
    Alternative to LayerNorm that works better for RL.
    """

    def __init__(self, dim, simplex_dim=8, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.simplex_dim = simplex_dim
        self.temperature = temperature

    def forward(self, x):
        # Input: x of shape (batch, dim)
        # Output: normalized x of shape (batch, dim)

        # Check if input shape is valid and if dim is divisible by simplex_dim
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"Input x must have shape (batch, {self.dim}), got {tuple(x.shape)}"
            )
        if self.dim % self.simplex_dim != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by simplex_dim ({self.simplex_dim})"
            )

        # (batch, group, simplex_dim)
        group = self.dim // self.simplex_dim
        x = x.reshape(x.shape[0], group, self.simplex_dim)

        # per-simplex softmax normalization (SimNorm)
        x = F.softmax(x / self.temperature, dim=2)

        # flatten back to (batch, dim)
        x = x.reshape(x.shape[0], self.dim)

        return x
