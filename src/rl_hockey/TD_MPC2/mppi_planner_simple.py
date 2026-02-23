"""MPPI planner for TD-MPC2."""

import torch
import torch.nn as nn

from rl_hockey.TD_MPC2.util import gumbel_softmax_sample, two_hot_inv


class MPPIPlannerSimplePaper(nn.Module):
    def __init__(
        self,
        dynamics,
        reward,
        termination,
        q_ensemble,
        policy,
        horizon=5,
        num_samples=512,
        num_iterations=6,
        temperature=0.5,
        gamma=0.99,
        std_init=2.0,
        std_min=0.05,
        std_decay=0.9,
        num_bins=101,
        vmin=-100.0,
        vmax=100.0,
        num_pi_trajs=24,
        num_elites=64,
        opponent_norm=True,
        opponent_stratified_elites=True,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.reward = reward
        self.termination = termination
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
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.num_pi_trajs = num_pi_trajs
        self.num_elites = num_elites
        self.opponent_norm = opponent_norm
        self.opponent_stratified_elites = opponent_stratified_elites

        self.register_buffer(
            "gamma_powers",
            torch.tensor([gamma**h for h in range(horizon + 1)], dtype=torch.float32),
        )
        self.register_buffer("_prev_mean", None)

    def rollout_trajectories(
        self,
        z_init,
        actions,
        dynamics,
        reward_predictor,
        termination_predictor,
        gamma=0.99,
    ):
        """Rollout trajectories in latent space."""
        num_samples, horizon, action_dim = actions.shape

        z = z_init.unsqueeze(0).expand(num_samples, -1).clone()

        returns = torch.zeros(num_samples, device=z.device)
        termination_probs = torch.zeros(num_samples, device=z.device)

        if hasattr(self, "gamma_powers"):
            gamma_powers = self.gamma_powers[:horizon]
        else:
            gamma_powers = torch.tensor(
                [gamma**h for h in range(horizon)], device=z.device
            )

        for h in range(horizon):
            a = actions[:, h, :]
            r_logits = reward_predictor(z, a)
            r = two_hot_inv(r_logits, self.num_bins, self.vmin, self.vmax).squeeze(-1)
            returns += gamma_powers[h] * r * (1 - termination_probs)

            z = dynamics(z, a).clone()

            t_logits = termination_predictor(z)
            t_prob = torch.sigmoid(t_logits).squeeze(-1)

            termination_probs = torch.clip(termination_probs + t_prob, 0, 1)

        return returns, z, termination_probs

    def plan(self, latent, return_mean=True, t0=False, return_stats=False):
        """Plan action using MPPI."""
        if self.action_dim is None:
            if self.policy is not None:
                test_mean = self.policy.mean_action(
                    latent.unsqueeze(0) if latent.dim() == 1 else latent
                )
                if test_mean.dim() > 1:
                    test_mean = test_mean.squeeze(0)
                self.action_dim = test_mean.shape[-1]
            else:
                raise ValueError("action_dim not set and no policy provided")

        if self._prev_mean is None:
            self.register_buffer(
                "_prev_mean",
                torch.zeros(self.horizon, self.action_dim, device=latent.device),
            )

        if self.policy is not None and self.num_pi_trajs > 0:
            # Assign one opponent per pi-trajectory for coherent action generation
            if hasattr(self.dynamics, "assign_opponents_for_batch"):
                self.dynamics.assign_opponents_for_batch(
                    self.num_pi_trajs, device=latent.device, balanced=True
                )

            pi_actions = torch.empty(
                self.horizon, self.num_pi_trajs, self.action_dim, device=latent.device
            )
            _z = (
                latent.repeat(self.num_pi_trajs, 1)
                if latent.dim() == 1
                else latent.repeat(self.num_pi_trajs, 1)
            )
            for t in range(self.horizon - 1):
                pi_action, _, _, _ = self.policy.sample(_z)
                pi_actions[t] = pi_action
                _z = self.dynamics(_z, pi_actions[t]).clone()
            pi_action, _, _, _ = self.policy.sample(_z)
            pi_actions[-1] = pi_action

        mean = torch.zeros(self.horizon, self.action_dim, device=latent.device)
        std = torch.full(
            (self.horizon, self.action_dim),
            self.std_init,
            dtype=torch.float,
            device=latent.device,
        )

        if not t0 and self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]

        actions = torch.empty(
            self.horizon, self.num_samples, self.action_dim, device=latent.device
        )

        if self.policy is not None and self.num_pi_trajs > 0:
            actions[:, : self.num_pi_trajs] = pi_actions

        # Assign one opponent per trajectory, fixed for all MPPI iterations.
        # balanced=True ensures each opponent gets an equal share of trajectories,
        # preventing a numerically lucky opponent from dominating the batch by chance.
        if hasattr(self.dynamics, "assign_opponents_for_batch"):
            self.dynamics.assign_opponents_for_batch(
                self.num_samples, device=latent.device, balanced=True
            )

        # Cache opponent IDs for normalization/stratification inside the loop.
        # This is read once here since the assignment does not change across iterations.
        batch_opponent_ids = None
        unique_opps = None
        n_opps = 1
        if (
            hasattr(self.dynamics, "_batch_opponent_ids")
            and self.dynamics._batch_opponent_ids is not None
        ):
            batch_opponent_ids = self.dynamics._batch_opponent_ids
            unique_opps = batch_opponent_ids.unique()
            n_opps = len(unique_opps)

        planning_stats = {
            "elite_returns_per_iter": [],
            "std_per_iter": [],
            "mean_return_per_iter": [],
            "all_returns_per_iter": [],
        }

        for iteration in range(self.num_iterations):
            num_random_samples = self.num_samples - self.num_pi_trajs
            if num_random_samples > 0:
                r = torch.randn(
                    self.horizon, num_random_samples, self.action_dim, device=std.device
                )
                actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
                actions_sample = actions_sample.clamp(-1, 1)
                actions[:, self.num_pi_trajs :] = actions_sample

            actions_reshaped = actions.transpose(0, 1)

            returns, final_z, final_termination_probs = self.rollout_trajectories(
                latent,
                actions_reshaped,
                self.dynamics,
                self.reward,
                self.termination,
                self.gamma,
            )

            final_actions = self.policy.mean_action(final_z)
            q_values = self.q_ensemble.avg(final_z, final_actions)
            if hasattr(self, "gamma_powers"):
                terminal_gamma = self.gamma_powers[self.horizon]
            else:
                terminal_gamma = self.gamma**self.horizon
            returns += (
                terminal_gamma * q_values.squeeze(-1) * (1 - final_termination_probs)
            )

            returns = returns.nan_to_num(0)

            # Normalize returns within each opponent group so that trajectories
            # against an "easy" opponent (high absolute returns) do not crowd out
            # trajectories against a "hard" opponent (lower absolute returns).
            # After normalization every group is on the same z-score scale.
            returns_for_scoring = returns.clone()
            if self.opponent_norm and batch_opponent_ids is not None and n_opps > 1:
                for oid in unique_opps:
                    mask = batch_opponent_ids == oid
                    if mask.sum() > 1:
                        r_grp = returns_for_scoring[mask]
                        returns_for_scoring[mask] = (
                            r_grp - r_grp.mean()
                        ) / r_grp.std().clamp(min=1e-6)

            # Pick top (num_elites // N) from each opponent group so every opponent
            # contributes equally to the distribution update regardless of how its
            # normalized scores compare to the other group.
            if (
                self.opponent_stratified_elites
                and batch_opponent_ids is not None
                and n_opps > 1
            ):
                elites_per_opp = self.num_elites // n_opps
                remainder = self.num_elites - elites_per_opp * n_opps
                elite_idx_list = []
                for i, oid in enumerate(unique_opps.tolist()):
                    grp_indices = (batch_opponent_ids == oid).nonzero(as_tuple=True)[0]
                    k = elites_per_opp + (1 if i < remainder else 0)
                    k = min(k, len(grp_indices))
                    if k > 0:
                        top_local = torch.topk(
                            returns_for_scoring[grp_indices], k, dim=0
                        ).indices
                        elite_idx_list.append(grp_indices[top_local])
                elite_idxs = torch.cat(elite_idx_list)
            else:
                elite_idxs = torch.topk(
                    returns_for_scoring, self.num_elites, dim=0
                ).indices

            elite_returns = returns[elite_idxs]  # raw returns for stats
            elite_scores = returns_for_scoring[elite_idxs]  # normalized for weights
            elite_actions = actions[:, elite_idxs]

            if return_stats:
                planning_stats["elite_returns_per_iter"].append(
                    {
                        "min": elite_returns.min().item(),
                        "max": elite_returns.max().item(),
                        "mean": elite_returns.mean().item(),
                        "std": elite_returns.std().item(),
                    }
                )
                planning_stats["mean_return_per_iter"].append(returns.mean().item())
                planning_stats["all_returns_per_iter"].append(
                    {
                        "min": returns.min().item(),
                        "max": returns.max().item(),
                        "std": returns.std().item(),
                    }
                )
                planning_stats["std_per_iter"].append(std.mean().item())

            # MPPI weights use normalized scores so that temperature has a stable
            # meaning independent of the absolute return scale of any opponent.
            max_return = elite_scores.max()
            score = torch.exp(self.temperature * (elite_scores - max_return))
            score = score / (score.sum() + 1e-9)

            score_expanded = score.unsqueeze(0).unsqueeze(-1)
            mean = (score_expanded * elite_actions).sum(dim=1) / (score.sum() + 1e-9)

            mean_expanded = mean.unsqueeze(1)
            std = (
                (score_expanded * (elite_actions - mean_expanded) ** 2).sum(dim=1)
                / (score.sum() + 1e-9)
            ).sqrt()
            std = std.clamp(self.std_min, self.std_init)

        # Clear per-trajectory opponent assignments
        if hasattr(self.dynamics, "clear_batch_opponents"):
            self.dynamics.clear_batch_opponents()

        if self._prev_mean is not None:
            self._prev_mean.copy_(mean)

        rand_idx = gumbel_softmax_sample(score)
        action = elite_actions[0, rand_idx, :]
        final_std = std[0]

        if not return_mean:
            action = action + final_std * torch.randn(
                self.action_dim, device=action.device
            )
            action = action.clamp(-1, 1)

        if return_stats:
            planning_stats["final_elite_returns"] = {
                "min": elite_returns.min().item(),
                "max": elite_returns.max().item(),
                "mean": elite_returns.mean().item(),
                "std": elite_returns.std().item(),
            }
            planning_stats["final_mean"] = mean[0].clone()
            planning_stats["final_std"] = std[0].mean().item()
            planning_stats["std_convergence"] = (
                planning_stats["std_per_iter"][0] - planning_stats["std_per_iter"][-1]
                if len(planning_stats["std_per_iter"]) > 1
                else 0.0
            )
            return action, planning_stats

        return action
