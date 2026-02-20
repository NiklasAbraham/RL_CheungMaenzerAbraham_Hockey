import logging
import torch
import torch.nn as nn
from torch.distributions import Normal

logger = logging.getLogger(__name__)


# https://arxiv.org/pdf/1702.03275 (code adapted from torchrl)
class BatchRenorm1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        momentum: float = 0.01,
        eps: float = 1e-5,
        max_r: float = 3.0,
        max_d: float = 5.0,
        warmup_steps: int = 10000,
        smooth: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.max_r = max_r
        self.max_d = max_d
        self.warmup_steps = warmup_steps
        self.smooth = smooth

        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float32)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float32)
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.dim() >= 2:
            raise ValueError(
                f"The {type(self).__name__} expects a 2D (or more) tensor, got {x.dim()}."
            )

        view_dims = [1, x.shape[1]] + [1] * (x.dim() - 2)

        def _v(v):
            return v.view(view_dims)

        running_std = (self.running_var + self.eps).sqrt_()

        if self.training:
            reduce_dims = [i for i in range(x.dim()) if i != 1]
            b_mean = x.mean(reduce_dims)
            b_var = x.var(reduce_dims, unbiased=False)
            b_std = (b_var + self.eps).sqrt_()

            r = torch.clamp((b_std.detach() / running_std), 1 / self.max_r, self.max_r)
            d = torch.clamp(
                (b_mean.detach() - self.running_mean) / running_std,
                -self.max_d,
                self.max_d,
            )

            # Compute warmup factor (0 during warmup, 1 after warmup)
            if self.warmup_steps > 0:
                if self.smooth:
                    warmup_factor = self.num_batches_tracked / self.warmup_steps
                else:
                    warmup_factor = self.num_batches_tracked // self.warmup_steps
                r = 1.0 + (r - 1.0) * warmup_factor
                d = d * warmup_factor

            x = (x - _v(b_mean)) / _v(b_std) * _v(r) + _v(d)

            unbiased_var = b_var.detach() * x.shape[0] / (x.shape[0] - 1)
            self.running_var += self.momentum * (unbiased_var - self.running_var)
            self.running_mean += self.momentum * (b_mean.detach() - self.running_mean)
            self.num_batches_tracked += 1
            self.num_batches_tracked = self.num_batches_tracked.clamp_max(self.warmup_steps)
        else:
            x = (x - _v(self.running_mean)) / _v(running_std)

        x = _v(self.weight) * x + _v(self.bias)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dims, activation, use_batchnorm=False):
        super().__init__()

        layers = []
        dim = input_dim
        for latent_dim in latent_dims:
            layers.append(nn.Linear(dim, latent_dim))
            layers.append(activation())
            if use_batchnorm:
                layers.append(BatchRenorm1d(latent_dim))
            dim = latent_dim
        layers.append(nn.Linear(dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dims=[256, 256], activation=nn.ReLU, use_batchnorm=False):
        super().__init__()

        self.net = MLP(state_dim + action_dim, 1, latent_dims, activation, use_batchnorm=use_batchnorm)

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.net(input)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dims=[256, 256], activation=nn.ReLU, use_batchnorm=False):
        super().__init__()

        self.net = MLP(state_dim, 2 * action_dim, latent_dims, activation, use_batchnorm=use_batchnorm)

    def forward(self, state):
        output = self.net(state)
        mean, log_std = torch.chunk(output, 2, dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state, noise=None, calc_log_prob=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if noise is None:
            noise = torch.randn_like(mean)

        action = mean + std*noise
        action_squashed = torch.tanh(action)

        log_prob = None
        if calc_log_prob:
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action)
            log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6)    # correct for tanh squashing
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action_squashed, log_prob
