# TD-MPC2 implementation

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from rl_hockey.common.agent import Agent
from rl_hockey.common.buffer import TDMPC2ReplayBuffer
from rl_hockey.TD_MPC2.model_dynamics_simple import DynamicsSimple
from rl_hockey.TD_MPC2.model_encoder import Encoder
from rl_hockey.TD_MPC2.model_init import (
    init_dynamics,
    init_encoder,
    init_policy,
    init_q_ensemble,
    init_reward,
    init_termination,
)
from rl_hockey.TD_MPC2.model_policy import Policy
from rl_hockey.TD_MPC2.model_q_ensemble import QEnsemble
from rl_hockey.TD_MPC2.model_reward import Reward
from rl_hockey.TD_MPC2.model_termination import Termination
from rl_hockey.TD_MPC2.mppi_planner_simple import MPPIPlannerSimplePaper
from rl_hockey.TD_MPC2.util import RunningScale, soft_ce, two_hot_inv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class TDMPC2(Agent):
    def __init__(
        self,
        obs_dim=18,
        action_dim=8,
        latent_dim=512,
        hidden_dim=None,
        num_q=5,
        simnorm_temperature=1.0,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        enc_lr_scale=0.3,
        gamma=0.99,
        lambda_coef=0.5,
        entropy_coef=1e-4,
        horizon=3,
        num_samples=512,
        num_iterations=6,
        num_elites=64,
        num_pi_trajs=24,
        capacity=1000000,
        temperature=0.5,
        batch_size=256,
        device="cuda",
        num_bins=101,
        vmin=-10.0,
        vmax=10.0,
        tau=0.01,
        grad_clip_norm=20.0,
        consistency_coef=20.0,
        reward_coef=0.1,
        value_coef=0.1,
        termination_coef=0.5,
        n_step=1,
        win_reward_bonus=10.0,
        win_reward_discount=0.92,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Handle hidden_dim dict, filling in missing network types with defaults
        if hidden_dim is None:
            default_hidden_dim = [256, 256, 256]
            hidden_dim = {
                "encoder": default_hidden_dim,
                "dynamics": default_hidden_dim,
                "reward": default_hidden_dim,
                "termination": default_hidden_dim,
                "q_function": default_hidden_dim,
                "policy": default_hidden_dim,
            }

        if not isinstance(hidden_dim, dict):
            raise ValueError(
                f"hidden_dim must be a dict with network-specific hidden dimensions, got {type(hidden_dim)}"
            )

        default_hidden_dim = [256, 256, 256]
        self.hidden_dim = hidden_dim
        self.hidden_dim_dict = {
            "encoder": hidden_dim.get("encoder", default_hidden_dim),
            "dynamics": hidden_dim.get("dynamics", default_hidden_dim),
            "reward": hidden_dim.get("reward", default_hidden_dim),
            "termination": hidden_dim.get("termination", default_hidden_dim),
            "q_function": hidden_dim.get("q_function", default_hidden_dim),
            "policy": hidden_dim.get("policy", default_hidden_dim),
        }

        self.num_q = num_q
        self.simnorm_temperature = simnorm_temperature
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lr = lr
        self.enc_lr_scale = enc_lr_scale
        self.gamma = gamma
        self.lambda_coef = lambda_coef
        self.entropy_coef = entropy_coef
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.num_elites = num_elites
        self.num_pi_trajs = num_pi_trajs
        self.temperature = temperature
        self.capacity = capacity
        self.device = torch.device(device)
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.termination_coef = termination_coef
        self.n_step = n_step

        self.config = {
            "batch_size": batch_size,
            "learning_rate": lr,
            "gamma": gamma,
            "horizon": horizon,
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "temperature": temperature,
            "simnorm_temperature": simnorm_temperature,
            "log_std_min": log_std_min,
            "log_std_max": log_std_max,
            "lambda_coef": lambda_coef,
            "entropy_coef": entropy_coef,
            "num_bins": num_bins,
            "vmin": vmin,
            "vmax": vmax,
            "consistency_coef": consistency_coef,
            "reward_coef": reward_coef,
            "value_coef": value_coef,
            "termination_coef": termination_coef,
            "tau": tau,
            "grad_clip_norm": grad_clip_norm,
            "n_step": n_step,
        }

        self.encoder = Encoder(
            obs_dim,
            latent_dim,
            self.hidden_dim_dict["encoder"],
            simnorm_temperature=simnorm_temperature,
        ).to(self.device)
        self.dynamics = DynamicsSimple(
            latent_dim,
            action_dim,
            self.hidden_dim_dict["dynamics"],
            simnorm_temperature=simnorm_temperature,
        ).to(self.device)
        self.reward = Reward(
            latent_dim,
            action_dim,
            self.hidden_dim_dict["reward"],
            num_bins,
            vmin=vmin,
            vmax=vmax,
        ).to(self.device)
        self.termination = Termination(
            latent_dim, self.hidden_dim_dict["termination"]
        ).to(self.device)
        self.q_ensemble = QEnsemble(
            num_q,
            latent_dim,
            action_dim,
            self.hidden_dim_dict["q_function"],
            num_bins,
            vmin,
            vmax,
        ).to(self.device)

        self._init_detached_q_ensemble()

        self.target_q_ensemble = copy.deepcopy(self.q_ensemble)
        for param in self.target_q_ensemble.parameters():
            param.requires_grad = False
        self.policy = Policy(
            latent_dim,
            action_dim,
            self.hidden_dim_dict["policy"],
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        init_encoder(self.encoder)
        init_dynamics(self.dynamics)
        init_reward(self.reward)
        init_termination(self.termination)
        init_q_ensemble(self.q_ensemble)
        init_policy(self.policy)

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": self.lr * self.enc_lr_scale,
                },
                {"params": self.dynamics.parameters()},
                {"params": self.reward.parameters()},
                {"params": self.termination.parameters()},
                {"params": self.q_ensemble.parameters()},
            ],
            lr=self.lr,
        )
        self.pi_optimizer = torch.optim.Adam(
            list(self.policy.parameters()), lr=self.lr, eps=1e-5
        )

        self.scale = RunningScale(tau=self.tau).to(self.device)

        self.planner = MPPIPlannerSimplePaper(
            self.dynamics,
            self.reward,
            self.termination,
            self.target_q_ensemble,
            self.policy,
            horizon,
            num_samples,
            num_iterations,
            temperature,
            self.gamma,
            std_init=2.0,
            std_min=0.05,
            num_bins=num_bins,
            vmin=vmin,
            vmax=vmax,
            num_pi_trajs=num_pi_trajs,
            num_elites=num_elites,
        )
        self.planners = None

        self._model_params = (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward.parameters())
            + list(self.termination.parameters())
            + list(self.q_ensemble.parameters())
        )

        batch_size = self.config.get("batch_size", 256)
        self.buffer = TDMPC2ReplayBuffer(
            max_size=capacity,
            horizon=self.horizon,
            batch_size=batch_size,
            use_torch_tensors=True,
            device=self.device,
            win_reward_bonus=win_reward_bonus,
            win_reward_discount=win_reward_discount,
        )
        self._obs_buffer = torch.empty(
            (batch_size, obs_dim), device=self.device, dtype=torch.float32
        )
        self._actions_buffer = torch.empty(
            (batch_size, action_dim), device=self.device, dtype=torch.float32
        )
        self._rewards_buffer = torch.empty(
            (batch_size, 1), device=self.device, dtype=torch.float32
        )
        self._next_obs_buffer = torch.empty(
            (batch_size, obs_dim), device=self.device, dtype=torch.float32
        )
        self._dones_buffer = torch.empty(
            (batch_size, 1), device=self.device, dtype=torch.float32
        )

        compile_available = hasattr(torch, "compile")
        if compile_available:
            try:
                self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
                self.dynamics = torch.compile(self.dynamics, mode="reduce-overhead")
                self.reward = torch.compile(self.reward, mode="reduce-overhead")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed: {e}. Using eager mode for all models."
                )

        self.planner.to(self.device)

    def store_transition(self, transition, winner=None):
        """Store a transition in the replay buffer.

        Args:
            transition: (state, action, reward, next_state, done)
            winner: Optional winner information (1 for agent win, -1 for loss, 0 for draw).
                Used for reward shaping in winning episodes.
        """
        self.buffer.store(transition, winner=winner)

    @torch.no_grad()
    def rollout_dynamics_multi_step(self, z0, action_sequence, max_horizon):
        """
        Roll out the dynamics model from z0 using the given action sequence.
        Returns predicted latent states at 1, 2, ..., max_horizon steps ahead.

        z0: (latent_dim,) numpy or tensor
        action_sequence: list of (action_dim,) arrays, length >= max_horizon
        max_horizon: int

        Returns:
            dict[int, np.ndarray]: {1: z_1, 2: z_2, ..., max_horizon: z_h}
        """
        if isinstance(z0, np.ndarray):
            z = torch.FloatTensor(z0).to(self.device).unsqueeze(0)
        else:
            z = z0.to(self.device)
            if z.dim() == 1:
                z = z.unsqueeze(0)
        out = {}
        for h in range(1, max_horizon + 1):
            a = action_sequence[h - 1]
            if isinstance(a, np.ndarray):
                a = torch.FloatTensor(a).to(self.device).unsqueeze(0)
            else:
                a = a.to(self.device)
                if a.dim() == 1:
                    a = a.unsqueeze(0)
            z = self.dynamics(z, a)
            out[h] = z.squeeze(0).cpu().numpy()
        return out

    @torch.no_grad()
    def act(self, obs, deterministic=False, t0=False):
        """Select action using MPC planning."""
        obs = torch.FloatTensor(obs).to(self.device)

        z = self.encoder(obs.unsqueeze(0)).squeeze(0)

        action = self.planner.plan(z, return_mean=deterministic, t0=t0)

        return action.cpu().numpy()

    @torch.no_grad()
    def act_with_stats(
        self,
        obs,
        deterministic=False,
        prev_action=None,
        prev_latent=None,
        prev_predicted_next_latent=None,
        t0=False,
    ):
        """Select action using MPC planning and collect statistics."""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_batch = obs_tensor.unsqueeze(0)

        z = self.encoder(obs_batch).squeeze(0)
        z = z.clone()

        plan_result = self.planner.plan(
            z, return_mean=deterministic, return_stats=True, t0=t0
        )
        if isinstance(plan_result, tuple):
            action, planning_stats = plan_result
        else:
            action = plan_result
            planning_stats = {}

        action = action.clone()

        stats = {}

        stats["encoder_latent_norm"] = z.norm().item()
        stats["encoder_latent_mean"] = z.mean().item()
        stats["encoder_latent_std"] = z.std().item()
        stats["_latent_state"] = z.cpu().numpy()

        policy_action = self.policy.mean_action(z.unsqueeze(0)).squeeze(0)

        q_logits_list = self.q_ensemble(z.unsqueeze(0), action.unsqueeze(0))
        from rl_hockey.TD_MPC2.util import two_hot_inv

        q_values = []
        for q_logits in q_logits_list:
            q_val = two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax).item()
            q_values.append(q_val)

        stats["q_values"] = q_values
        stats["q_min"] = min(q_values)
        stats["q_max"] = max(q_values)
        stats["q_mean"] = sum(q_values) / len(q_values)
        q_std_val = torch.tensor(q_values).std().item() if len(q_values) > 1 else 0.0
        stats["q_std"] = q_std_val

        stats["q_spread"] = max(q_values) - min(q_values)
        q_mean_val = stats["q_mean"]
        stats["q_coefficient_of_variation"] = (
            q_std_val / abs(q_mean_val) if abs(q_mean_val) > 1e-8 else 0.0
        )

        q_logits_policy = self.q_ensemble(z.unsqueeze(0), policy_action.unsqueeze(0))
        q_values_policy = []
        for q_logits in q_logits_policy:
            q_val = two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax).item()
            q_values_policy.append(q_val)

        stats["q_policy_values"] = q_values_policy
        stats["q_policy_min"] = min(q_values_policy)
        stats["q_policy_max"] = max(q_values_policy)
        stats["q_policy_mean"] = sum(q_values_policy) / len(q_values_policy)

        z_next_pred = self.dynamics(z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
        latent_change = z_next_pred - z
        stats["dynamics_latent_change_norm"] = latent_change.norm().item()
        stats["dynamics_latent_change_mean"] = latent_change.mean().item()
        stats["dynamics_latent_change_std"] = latent_change.std().item()
        stats["dynamics_latent_next_norm"] = z_next_pred.norm().item()

        if prev_predicted_next_latent is not None:
            prev_pred_tensor = (
                prev_predicted_next_latent
                if isinstance(prev_predicted_next_latent, torch.Tensor)
                else torch.FloatTensor(prev_predicted_next_latent).to(self.device)
            )
            dynamics_error = z - prev_pred_tensor
            stats["dynamics_prediction_error_norm"] = dynamics_error.norm().item()
            stats["dynamics_prediction_error_mse"] = (dynamics_error**2).mean().item()
            stats["dynamics_prediction_error_mean"] = dynamics_error.mean().item()
            stats["dynamics_prediction_error_std"] = dynamics_error.std().item()
        else:
            stats["dynamics_prediction_error_norm"] = None
            stats["dynamics_prediction_error_mse"] = None
            stats["dynamics_prediction_error_mean"] = None
            stats["dynamics_prediction_error_std"] = None

        stats["_predicted_next_latent"] = z_next_pred.clone().cpu().numpy()

        reward_logits = self.reward(z.unsqueeze(0), action.unsqueeze(0))
        reward_pred = two_hot_inv(
            reward_logits, self.num_bins, self.vmin, self.vmax
        ).item()
        stats["reward_pred"] = reward_pred

        stats["action_norm"] = action.norm().item()
        stats["action_mean"] = action.mean().item()
        stats["action_std"] = action.std().item()
        stats["action_min"] = action.min().item()
        stats["action_max"] = action.max().item()

        action_diff = action - policy_action
        stats["action_policy_diff_norm"] = action_diff.norm().item()
        stats["action_policy_diff_mean"] = action_diff.mean().item()

        if prev_action is not None:
            prev_action_tensor = torch.FloatTensor(prev_action).to(self.device)
            action_diff_temporal = action - prev_action_tensor
            stats["action_smoothness"] = action_diff_temporal.norm().item()
        else:
            stats["action_smoothness"] = None

        if prev_latent is not None:
            prev_latent_tensor = (
                prev_latent
                if isinstance(prev_latent, torch.Tensor)
                else torch.FloatTensor(prev_latent).to(self.device)
            )
            latent_diff_temporal = z - prev_latent_tensor
            stats["latent_smoothness"] = latent_diff_temporal.norm().item()
        else:
            stats["latent_smoothness"] = None

        if planning_stats:
            stats["planning_stats"] = planning_stats
            if "final_elite_returns" in planning_stats:
                final_elite = planning_stats["final_elite_returns"]
                stats["mppi_elite_return_min"] = final_elite["min"]
                stats["mppi_elite_return_max"] = final_elite["max"]
                stats["mppi_elite_return_mean"] = final_elite["mean"]
                stats["mppi_elite_return_std"] = final_elite["std"]
            if "final_std" in planning_stats:
                stats["mppi_final_std"] = planning_stats["final_std"]
            if "std_convergence" in planning_stats:
                stats["mppi_std_convergence"] = planning_stats["std_convergence"]

        return action.cpu().numpy(), stats

    @torch.no_grad()
    def act_batch(self, obs_batch, deterministic=False, t0s=None):
        """Select actions for batch of observations."""
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        num_envs = obs_batch.shape[0]

        if self.planners is None or len(self.planners) != num_envs:
            self.planners = [copy.deepcopy(self.planner) for _ in range(num_envs)]

        z_batch = self.encoder(obs_batch)
        actions = []
        if t0s is None:
            t0s = [False] * num_envs

        for i, (z, t0) in enumerate(zip(z_batch, t0s)):
            action = self.planners[i].plan(z, return_mean=deterministic, t0=t0)
            actions.append(action)

        return torch.stack(actions).cpu().numpy()

    def evaluate(self, obs):
        """Evaluate state value using Q-ensemble."""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            z = self.encoder(obs.unsqueeze(0))
            action = self.policy.mean_action(z)
            q_value = self.q_ensemble.min(z, action)

            return q_value.item()

    def train(self, steps=1):
        all_losses = {
            "consistency_loss": [],
            "reward_loss": [],
            "value_loss": [],
            "termination_loss": [],
            "policy_loss": [],
            "total_loss": [],
            "loss": [],
            "grad_norm_encoder": [],
            "grad_norm_dynamics": [],
            "grad_norm_reward": [],
            "grad_norm_termination": [],
            "grad_norm_q_ensemble": [],
            "grad_norm_policy": [],
        }

        for _ in range(steps):
            batch_size = self.config.get("batch_size", 256)

            used_sequences = False
            if hasattr(self.buffer, "sample_sequences") and self.horizon > 1:
                seq = self.buffer.sample_sequences(batch_size, self.horizon)
                if seq is not None:
                    obs_seq, actions_seq, rewards_seq, dones_seq = seq

                    if not isinstance(obs_seq, torch.Tensor):
                        obs_seq = torch.from_numpy(obs_seq).float()
                        actions_seq = torch.from_numpy(actions_seq).float()
                        rewards_seq = torch.from_numpy(rewards_seq).float()
                        dones_seq = torch.from_numpy(dones_seq).float()

                    obs_seq = obs_seq.to(self.device, non_blocking=True)
                    actions_seq = actions_seq.to(self.device, non_blocking=True)
                    rewards_seq = rewards_seq.to(self.device, non_blocking=True)
                    dones_seq = dones_seq.to(self.device, non_blocking=True)

                    if rewards_seq.dim() == 2:
                        rewards_seq = rewards_seq.unsqueeze(-1)
                    if dones_seq.dim() == 2:
                        dones_seq = dones_seq.unsqueeze(-1)

                    batch_size_actual, horizon_plus_one, _ = obs_seq.shape
                    horizon = horizon_plus_one - 1

                    obs_flat = obs_seq.reshape(batch_size_actual * (horizon + 1), -1)
                    z_flat = self.encoder(obs_flat)
                    z_seq = z_flat.reshape(
                        batch_size_actual, horizon + 1, self.latent_dim
                    )

                    zs = torch.empty(
                        horizon + 1,
                        batch_size_actual,
                        self.latent_dim,
                        device=self.device,
                    )
                    z_pred = z_seq[:, 0].clone()
                    zs[0] = z_pred
                    consistency_loss = 0.0
                    reward_loss = 0.0
                    value_loss = 0.0
                    for t in range(horizon):
                        a_t = actions_seq[:, t]
                        r_t = rewards_seq[:, t]
                        d_t = dones_seq[:, t]

                        z_next_pred = self.dynamics(z_pred, a_t)
                        z_target = z_seq[:, t + 1].detach()
                        zs[t + 1] = z_next_pred

                        weight = self.lambda_coef**t
                        consistency_loss = consistency_loss + weight * F.mse_loss(
                            z_next_pred, z_target
                        )

                        r_pred_logits = self.reward(z_pred, a_t)
                        reward_loss = (
                            reward_loss
                            + weight
                            * soft_ce(
                                r_pred_logits, r_t, self.num_bins, self.vmin, self.vmax
                            ).mean()
                        )

                        q_preds_logits = self.q_ensemble(z_pred, a_t)
                        with torch.no_grad():
                            n = min(self.n_step, horizon - t)
                            gamma_powers = (
                                self.gamma
                                ** torch.arange(
                                    n,
                                    device=rewards_seq.device,
                                    dtype=rewards_seq.dtype,
                                )
                            ).view(1, n, 1)
                            reward_sum = (rewards_seq[:, t : t + n] * gamma_powers).sum(
                                dim=1
                            )
                            z_bootstrap = z_seq[:, t + n].detach()
                            next_action, _, _, _ = self.policy.sample(z_bootstrap)
                            target_q = self.target_q_ensemble.min(
                                z_bootstrap, next_action
                            )
                            d_n = dones_seq[:, t + n - 1]
                            bootstrap = (self.gamma**n) * (1.0 - d_n) * target_q
                            q_target = reward_sum + bootstrap

                        step_value_loss = 0.0
                        for q_pred_logits in q_preds_logits:
                            step_value_loss = (
                                step_value_loss
                                + soft_ce(
                                    q_pred_logits,
                                    q_target,
                                    self.num_bins,
                                    self.vmin,
                                    self.vmax,
                                ).mean()
                            )
                        value_loss = value_loss + weight * step_value_loss

                        z_pred = z_next_pred

                    consistency_loss = consistency_loss / horizon
                    reward_loss = reward_loss / horizon
                    value_loss = value_loss / (horizon * self.num_q)
                    zs_bh = zs[1:].permute(1, 0, 2).reshape(-1, self.latent_dim)
                    termination_pred_logits = self.termination(zs_bh)
                    termination_target = dones_seq.reshape(-1, 1)
                    termination_loss = F.binary_cross_entropy_with_logits(
                        termination_pred_logits, termination_target
                    )

                    used_sequences = True

            if not used_sequences:
                raise RuntimeError(
                    "TD-MPC2 requires sequence sampling! "
                    "ReplayBuffer.sample_sequences returned None or is missing. "
                    "The agent cannot learn a consistent world model without sequences."
                )

            total_loss = (
                self.consistency_coef * consistency_loss
                + self.reward_coef * reward_loss
                + self.value_coef * value_loss
                + self.termination_coef * termination_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()

            # Compute gradient norms before clipping
            grad_norm_encoder = torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=float("inf")
            )
            grad_norm_dynamics = torch.nn.utils.clip_grad_norm_(
                self.dynamics.parameters(), max_norm=float("inf")
            )
            grad_norm_reward = torch.nn.utils.clip_grad_norm_(
                self.reward.parameters(), max_norm=float("inf")
            )
            grad_norm_termination = torch.nn.utils.clip_grad_norm_(
                self.termination.parameters(), max_norm=float("inf")
            )
            grad_norm_q_ensemble = torch.nn.utils.clip_grad_norm_(
                self.q_ensemble.parameters(), max_norm=float("inf")
            )

            torch.nn.utils.clip_grad_norm_(
                self._model_params, max_norm=self.grad_clip_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Mark step boundary for CUDAGraphs to enable fast path
            # This ensures all backward operations are complete before next forward pass
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            policy_loss, grad_norm_policy = self._update_policy(zs.detach())
            self._update_target_network(tau=self.tau)
            if steps == 1:
                all_losses["consistency_loss"].append(consistency_loss.item())
                all_losses["reward_loss"].append(reward_loss.item())
                all_losses["value_loss"].append(value_loss.item())
                all_losses["termination_loss"].append(termination_loss.item())
                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["total_loss"].append(total_loss.item())
                all_losses["loss"].append(total_loss.item())
                all_losses["grad_norm_encoder"].append(grad_norm_encoder.item())
                all_losses["grad_norm_dynamics"].append(grad_norm_dynamics.item())
                all_losses["grad_norm_reward"].append(grad_norm_reward.item())
                all_losses["grad_norm_termination"].append(grad_norm_termination.item())
                all_losses["grad_norm_q_ensemble"].append(grad_norm_q_ensemble.item())
                all_losses["grad_norm_policy"].append(grad_norm_policy.item())
            else:
                all_losses["consistency_loss"].append(consistency_loss)
                all_losses["reward_loss"].append(reward_loss)
                all_losses["value_loss"].append(value_loss)
                all_losses["termination_loss"].append(termination_loss)
                all_losses["policy_loss"].append(policy_loss)
                all_losses["total_loss"].append(total_loss)
                all_losses["loss"].append(total_loss)
                all_losses["grad_norm_encoder"].append(grad_norm_encoder)
                all_losses["grad_norm_dynamics"].append(grad_norm_dynamics)
                all_losses["grad_norm_reward"].append(grad_norm_reward)
                all_losses["grad_norm_termination"].append(grad_norm_termination)
                all_losses["grad_norm_q_ensemble"].append(grad_norm_q_ensemble)
                all_losses["grad_norm_policy"].append(grad_norm_policy)

        if steps > 1 and len(all_losses["consistency_loss"]) > 0:
            if isinstance(all_losses["consistency_loss"][0], torch.Tensor):
                all_losses["consistency_loss"] = [
                    loss.item() for loss in all_losses["consistency_loss"]
                ]
                all_losses["reward_loss"] = [
                    loss.item() for loss in all_losses["reward_loss"]
                ]
                all_losses["value_loss"] = [
                    loss.item() for loss in all_losses["value_loss"]
                ]
                all_losses["termination_loss"] = [
                    loss.item() for loss in all_losses["termination_loss"]
                ]
                all_losses["policy_loss"] = [
                    loss.item() for loss in all_losses["policy_loss"]
                ]
                all_losses["total_loss"] = [
                    loss.item() for loss in all_losses["total_loss"]
                ]
                all_losses["loss"] = [loss.item() for loss in all_losses["loss"]]
                all_losses["grad_norm_encoder"] = [
                    grad.item() for grad in all_losses["grad_norm_encoder"]
                ]
                all_losses["grad_norm_dynamics"] = [
                    grad.item() for grad in all_losses["grad_norm_dynamics"]
                ]
                all_losses["grad_norm_reward"] = [
                    grad.item() for grad in all_losses["grad_norm_reward"]
                ]
                all_losses["grad_norm_termination"] = [
                    grad.item() for grad in all_losses["grad_norm_termination"]
                ]
                all_losses["grad_norm_q_ensemble"] = [
                    grad.item() for grad in all_losses["grad_norm_q_ensemble"]
                ]
                all_losses["grad_norm_policy"] = [
                    grad.item() for grad in all_losses["grad_norm_policy"]
                ]

        return all_losses

    def _init_detached_q_ensemble(self):
        """
        Initialize detached Q-ensemble.

        Detachment is handled via stateless functional_call in _update_policy,
        so this is intentionally a no-op aside from keeping the API stable.
        """
        return

    def _q_ensemble_detached(self, latent, action):
        """Forward through Q-ensemble with detached parameters."""
        params = {
            name: param.detach() for name, param in self.q_ensemble.named_parameters()
        }
        buffers = dict(self.q_ensemble.named_buffers())
        return functional_call(self.q_ensemble, {**params, **buffers}, (latent, action))

    def _update_policy(self, zs):
        """Update policy to maximize Q-values + entropy."""
        self.pi_optimizer.zero_grad()

        # Mark step boundary for CUDAGraphs before forward pass
        # This helps CUDAGraphs use the fast path by ensuring clean state
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        num_states = zs.shape[0]
        batch_size = zs.shape[1]
        zs_flat = zs.reshape(-1, zs.shape[-1])
        actions, log_probs, _, scaled_entropies = self.policy.sample(zs_flat)
        q_logits_all = self._q_ensemble_detached(zs_flat, actions)

        q_values = torch.stack(
            [
                two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax)
                for q_logits in q_logits_all
            ],
            dim=0,
        )
        qs_flat = q_values.mean(dim=0)
        qs = qs_flat.reshape(num_states, batch_size, 1)

        if scaled_entropies.dim() == 1:
            scaled_entropies = scaled_entropies.reshape(num_states, batch_size, 1)
        else:
            scaled_entropies = scaled_entropies.reshape(num_states, batch_size, -1)

        self.scale.update(qs[0].detach())
        qs_scaled = self.scale(qs)
        rho = torch.pow(
            torch.tensor(self.lambda_coef, device=self.device),
            torch.arange(num_states, device=self.device),
        )
        pi_loss = (
            -(self.entropy_coef * scaled_entropies + qs_scaled).mean(dim=(1, 2)) * rho
        ).mean()

        pi_loss.backward()

        # Compute gradient norm before clipping
        grad_norm_policy = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=float("inf")
        )

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=self.grad_clip_norm
        )
        self.pi_optimizer.step()
        self.pi_optimizer.zero_grad(set_to_none=True)

        # Mark step boundary after policy update completes
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        return pi_loss.detach(), grad_norm_policy

    def _update_target_network(self, tau=0.01):
        with torch.no_grad():
            source_params = list(self.q_ensemble.parameters())
            target_params = list(self.target_q_ensemble.parameters())
            for param, target_param in zip(source_params, target_params):
                target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)

    def save(self, path):
        """Save agent"""
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "dynamics": self.dynamics.state_dict(),
                "reward": self.reward.state_dict(),
                "termination": self.termination.state_dict(),
                "q_ensemble": self.q_ensemble.state_dict(),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "pi_optimizer": self.pi_optimizer.state_dict(),
                "scale": self.scale.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "num_q": self.num_q,
                "num_bins": self.num_bins,
                "vmin": self.vmin,
                "vmax": self.vmax,
            },
            path,
        )

    def load(self, path):
        """Load agent"""
        checkpoint = torch.load(path, map_location="cpu")

        checkpoint_num_bins = checkpoint.get("num_bins") or checkpoint.get(
            "config", {}
        ).get("num_bins")
        if checkpoint_num_bins is not None and checkpoint_num_bins != self.num_bins:
            logger.warning(
                f"Architecture mismatch detected: checkpoint has num_bins={checkpoint_num_bins}, "
                f"but current model has num_bins={self.num_bins}. "
                f"Reward and Q-function output layers will be initialized from scratch."
            )

        checkpoint_latent_dim = checkpoint.get("latent_dim")
        checkpoint_action_dim = checkpoint.get("action_dim")
        if (
            checkpoint_latent_dim is not None
            and checkpoint_latent_dim != self.latent_dim
        ):
            raise ValueError(
                f"Cannot load checkpoint: latent_dim mismatch "
                f"(checkpoint: {checkpoint_latent_dim}, current: {self.latent_dim})"
            )
        if (
            checkpoint_action_dim is not None
            and checkpoint_action_dim != self.action_dim
        ):
            raise ValueError(
                f"Cannot load checkpoint: action_dim mismatch "
                f"(checkpoint: {checkpoint_action_dim}, current: {self.action_dim})"
            )

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])

        if checkpoint_num_bins is not None and checkpoint_num_bins != self.num_bins:
            missing_keys, unexpected_keys = self.reward.load_state_dict(
                checkpoint["reward"], strict=False
            )
            if missing_keys:
                logger.debug(f"Reward model missing keys (expected): {missing_keys}")
            if unexpected_keys:
                logger.debug(f"Reward model unexpected keys: {unexpected_keys}")
            logger.info(
                f"Reward model loaded with architecture adaptation "
                f"(num_bins: {checkpoint_num_bins} -> {self.num_bins}). "
                f"Output layer initialized from scratch."
            )
        else:
            self.reward.load_state_dict(checkpoint["reward"])

        if "termination" in checkpoint:
            self.termination.load_state_dict(checkpoint["termination"])
        else:
            logger.warning(
                "Termination model not found in checkpoint, initializing from scratch."
            )

        if checkpoint_num_bins is not None and checkpoint_num_bins != self.num_bins:
            missing_keys, unexpected_keys = self.q_ensemble.load_state_dict(
                checkpoint["q_ensemble"], strict=False
            )
            if missing_keys:
                logger.debug(f"Q ensemble missing keys (expected): {missing_keys}")
            if unexpected_keys:
                logger.debug(f"Q ensemble unexpected keys: {unexpected_keys}")
            logger.info(
                f"Q ensemble loaded with architecture adaptation "
                f"(num_bins: {checkpoint_num_bins} -> {self.num_bins}). "
                f"Output layers initialized from scratch."
            )
        else:
            self.q_ensemble.load_state_dict(checkpoint["q_ensemble"])

        self.policy.load_state_dict(checkpoint["policy"])

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

        if "scale" in checkpoint:
            try:
                self.scale.load_state_dict(checkpoint["scale"])
            except Exception as e:
                logger.warning(f"Could not load RunningScale state: {e}")

        if checkpoint_num_bins is not None and checkpoint_num_bins != self.num_bins:
            missing_keys, unexpected_keys = self.target_q_ensemble.load_state_dict(
                checkpoint["q_ensemble"], strict=False
            )
            if missing_keys:
                logger.debug(
                    f"Target Q ensemble missing keys (expected): {missing_keys}"
                )
            if unexpected_keys:
                logger.debug(f"Target Q ensemble unexpected keys: {unexpected_keys}")
        else:
            self.target_q_ensemble.load_state_dict(checkpoint["q_ensemble"])

        self._init_detached_q_ensemble()

    def log_architecture(self):
        """Log network architecture."""

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines = []
        lines.append("=" * 80)
        lines.append("TD-MPC2 AGENT ARCHITECTURE")
        lines.append("=" * 80)
        lines.append(f"Observation Dimension: {self.obs_dim}")
        lines.append(f"Action Dimension: {self.action_dim}")
        lines.append(f"Latent Dimension: {self.latent_dim}")
        lines.append("Hidden Dimensions (per network):")
        for network_type, hidden_dims in self.hidden_dim_dict.items():
            lines.append(f"  {network_type}: {hidden_dims}")
        lines.append(f"Device: {self.device}")
        lines.append("")

        lines.append("Configuration:")
        for key, value in self.config.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        lines.append("MODEL-BASED RL COMPONENTS:")
        lines.append("")

        lines.append("1. ENCODER NETWORK:")
        lines.append("   Maps observations to latent state representations")
        lines.append(str(self.encoder))
        lines.append(f"   Trainable Parameters: {count_parameters(self.encoder):,}")
        lines.append("")

        lines.append("2. DYNAMICS MODEL:")
        lines.append("   Predicts next latent state given current state and action")
        lines.append(str(self.dynamics))
        lines.append(f"   Trainable Parameters: {count_parameters(self.dynamics):,}")
        lines.append("")

        lines.append("3. REWARD MODEL:")
        lines.append("   Predicts reward given latent state and action")
        lines.append(str(self.reward))
        lines.append(f"   Trainable Parameters: {count_parameters(self.reward):,}")
        lines.append("")

        lines.append("3a. TERMINATION MODEL:")
        lines.append(
            "   Predicts termination probability given latent state and action"
        )
        lines.append(str(self.termination))
        lines.append(f"   Trainable Parameters: {count_parameters(self.termination):,}")
        lines.append("")

        lines.append("4. Q ENSEMBLE:")
        lines.append(f"   {self.num_q} Q-networks for value estimation")
        lines.append(str(self.q_ensemble))
        lines.append(f"   Trainable Parameters: {count_parameters(self.q_ensemble):,}")
        lines.append("")

        lines.append("5. TARGET Q ENSEMBLE:")
        lines.append("   Target network for stable Q-learning")
        lines.append("   Same architecture as Q Ensemble")
        lines.append(
            f"   Trainable Parameters: {count_parameters(self.target_q_ensemble):,}"
        )
        lines.append("")

        lines.append("6. POLICY NETWORK:")
        lines.append("   Learns to mimic the MPC planner for fast inference")
        lines.append(str(self.policy))
        lines.append(f"   Trainable Parameters: {count_parameters(self.policy):,}")
        lines.append("")

        model_params = (
            count_parameters(self.encoder)
            + count_parameters(self.dynamics)
            + count_parameters(self.reward)
            + count_parameters(self.termination)
            + count_parameters(self.q_ensemble)
        )
        total_params = (
            model_params
            + count_parameters(self.policy)
            + count_parameters(self.target_q_ensemble)
        )

        lines.append("PARAMETER SUMMARY:")
        lines.append(
            f"  World Model (Encoder + Dynamics + Reward + Termination + Q): {model_params:,}"
        )
        lines.append(f"  Policy Network: {count_parameters(self.policy):,}")
        lines.append(
            f"  Target Q Network: {count_parameters(self.target_q_ensemble):,}"
        )
        lines.append(f"  TOTAL TRAINABLE PARAMETERS: {total_params:,}")
        lines.append("")

        lines.append("OPTIMIZERS:")
        lines.append(
            f"  World Model + Q Optimizer: {self.optimizer.__class__.__name__} (LR: {self.lr})"
        )
        lines.append(
            f"  Policy Optimizer: {self.pi_optimizer.__class__.__name__} (LR: {self.lr})"
        )
        lines.append("")

        lines.append("MPC PLANNER:")
        lines.append("  Type: MPPI (Model Predictive Path Integral)")
        lines.append(f"  Horizon: {self.horizon}")
        lines.append(f"  Samples per iteration: {self.num_samples}")
        lines.append(f"  Planning iterations: {self.num_iterations}")
        lines.append(f"  Temperature: {self.temperature}")
        if hasattr(self, "fast_mode"):
            lines.append(f"  Fast Mode: {self.fast_mode} (use policy network only)")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
