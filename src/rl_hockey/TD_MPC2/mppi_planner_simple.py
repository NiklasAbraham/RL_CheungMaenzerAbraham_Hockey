# MPPI planner for simple from the TD-MPC2 paper.

import torch
import torch.nn as nn


class MPPIPlannerSimplePaper(nn.Module):
    def __init__(
        self,
        dynamics,
        reward,
        q_ensemble,
        policy,
        horizon=5,
        num_samples=512,
        num_iterations=6,
        temperature=0.5,
        gamma=0.99,
        std_init=0.5,
        std_min=0.05,
        std_decay=0.9,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.reward = reward
        self.q_ensemble = q_ensemble
        self.policy = policy
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.gamma = gamma
        self.std_init = std_init
        self.std_min = std_min
        self.std_decay = std_decay
        self.action_dim = None

        # Pre-compute gamma powers for efficiency (register as buffer so it moves with device)
        self.register_buffer(
            "gamma_powers",
            torch.tensor([gamma**h for h in range(horizon + 1)], dtype=torch.float32),
        )

    def sample_trajectories(self, mean, std, num_samples, horizon, action_dim):
        """
        Sample action trajectories from Gaussian distribution.

        Args:
            mean: (horizon, action_dim) mean actions
            std: scalar or (horizon, action_dim) standard deviation
            num_samples: number of trajectories to sample
            horizon: planning horizon
            action_dim: action dimension

        Returns:
            actions: (num_samples, horizon, action_dim)
        """
        noise = torch.randn(num_samples, horizon, action_dim, device=mean.device) * std
        actions = mean.unsqueeze(0) + noise
        actions = torch.clamp(actions, -1, 1)
        return actions

    def rollout_trajectories(
        self, z_init, actions, dynamics, reward_predictor, gamma=0.99
    ):
        """
        Rollout trajectories in latent space.

        Args:
            z_init: (latent_dim,) initial latent state
            actions: (num_samples, horizon, action_dim) action sequences
            dynamics: dynamics model
            reward_predictor: reward predictor
            gamma: discount factor

        Returns:
            returns: (num_samples,) discounted returns for each trajectory
            final_states: (num_samples, latent_dim) final latent states
        """
        num_samples, horizon, action_dim = actions.shape

        z = z_init.unsqueeze(0).expand(num_samples, -1)

        returns = torch.zeros(num_samples, device=z.device)

        # Use pre-computed gamma powers (automatically on correct device via register_buffer)
        if hasattr(self, "gamma_powers"):
            gamma_powers = self.gamma_powers[:horizon]
        else:
            gamma_powers = torch.tensor(
                [gamma**h for h in range(horizon)], device=z.device
            )

        for h in range(horizon):
            a = actions[:, h, :]

            r = reward_predictor(z, a).squeeze(-1)
            returns += gamma_powers[h] * r

            # Clone to avoid inplace modification issues (even though used with no_grad)
            z = dynamics(z, a).clone()

        return returns, z

    def plan(self, latent, return_mean=True):
        """
        Plan action using MPPI.
        """
        if self.policy is not None:
            mean = self.policy.mean_action(
                latent.unsqueeze(0) if latent.dim() == 1 else latent
            )
            if mean.dim() > 1:
                mean = mean.squeeze(0)
            if self.action_dim is None:
                self.action_dim = mean.shape[-1]
            mean = mean.unsqueeze(0).expand(self.horizon, -1)
        else:
            if self.action_dim is None:
                raise ValueError("action_dim not set and no policy provided")
            mean = torch.zeros(self.horizon, self.action_dim, device=latent.device)

        std = self.std_init

        for iteration in range(self.num_iterations):
            actions = self.sample_trajectories(
                mean, std, self.num_samples, self.horizon, self.action_dim
            )

            returns, final_z = self.rollout_trajectories(
                latent, actions, self.dynamics, self.reward, self.gamma
            )

            # Add terminal Q-value
            # Use policy action for terminal state z_{t+H}
            final_actions = self.policy.mean_action(final_z)
            q_values = self.q_ensemble.min(final_z, final_actions)
            # Use pre-computed gamma power (automatically on correct device)
            if hasattr(self, "gamma_powers"):
                terminal_gamma = self.gamma_powers[self.horizon]
            else:
                terminal_gamma = self.gamma**self.horizon
            returns += terminal_gamma * q_values.squeeze(-1)

            # Compute weights using softmax
            weights = torch.softmax(returns / self.temperature, dim=0)
            weights = weights.view(-1, 1, 1)
            mean = (weights * actions).sum(dim=0)

            # Decay std
            std = max(std * self.std_decay, self.std_min)

        if return_mean:
            return mean[0]
        else:
            best_idx = torch.argmax(weights.squeeze())
            return actions[best_idx, 0, :]
