# Utility functions for TD-MPC2.
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Simplicial normalization layer."""

    def __init__(self, dim, simplex_dim=8, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.simplex_dim = simplex_dim
        self.temperature = temperature

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"Input x must have shape (batch, {self.dim}), got {tuple(x.shape)}"
            )
        if self.dim % self.simplex_dim != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by simplex_dim ({self.simplex_dim})"
            )

        group = self.dim // self.simplex_dim
        x = x.reshape(x.shape[0], group, self.simplex_dim)
        x = F.softmax(x / self.temperature, dim=2)
        x = x.reshape(x.shape[0], self.dim)

        return x


def symlog(x):
    """Symmetric logarithmic function."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """Symmetric exponential function."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, num_bins, vmin, vmax):
    """Converts scalars to soft two-hot encoded targets."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    
    x_symlog = torch.clamp(symlog(x), vmin, vmax).squeeze(1)
    bin_size = (vmax - vmin) / (num_bins - 1)
    bin_idx = torch.floor((x_symlog - vmin) / bin_size)
    bin_offset = ((x_symlog - vmin) / bin_size - bin_idx).unsqueeze(-1)
    
    bin_idx = bin_idx.long()
    
    soft_two_hot = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(
        1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset
    )
    
    return soft_two_hot


def two_hot_inv(x, num_bins, vmin, vmax):
    """Converts soft two-hot encoded vectors to scalars."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symexp(x)
    
    x_probs = F.softmax(x, dim=-1)
    dreg_bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x_symlog = torch.sum(x_probs * dreg_bins, dim=-1, keepdim=True)
    return symexp(x_symlog)


def soft_ce(pred, target, num_bins, vmin, vmax):
    """Computes cross entropy loss between predictions and soft targets."""
    target_two_hot = two_hot(target, num_bins, vmin, vmax)
    pred_log_softmax = F.log_softmax(pred, dim=-1)
    loss = -(target_two_hot * pred_log_softmax).sum(-1, keepdim=True)
    
    return loss


class RunningScale(torch.nn.Module):
    """Running trimmed scale estimator for normalizing Q-values."""

    def __init__(self, tau=0.01):
        super().__init__()
        self.tau = tau
        self.register_buffer("value", torch.ones(1, dtype=torch.float32))
        self.register_buffer("_percentiles", torch.tensor([5, 95], dtype=torch.float32))

    def _positions(self, x_shape):
        positions = self._percentiles * (x_shape - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.flatten(1, x.ndim - 1) if x.ndim > 1 else x.unsqueeze(0)
        in_sorted = torch.sort(x, dim=0).values
        floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0])
        d0 = in_sorted[floored] * weight_floored
        d1 = in_sorted[ceiled] * weight_ceiled
        return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self.value.data.lerp_(value, self.tau)

    def forward(self, x):
        return x / self.value

    def __repr__(self):
        return f"RunningScale(value={self.value.item():.4f})"


def gumbel_softmax_sample(logits, temperature=1.0):
    """Sample from the Gumbel-Softmax distribution."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits.log() + gumbels) / temperature
    y_soft = gumbels.softmax(dim=-1)
    return y_soft.argmax(-1)
