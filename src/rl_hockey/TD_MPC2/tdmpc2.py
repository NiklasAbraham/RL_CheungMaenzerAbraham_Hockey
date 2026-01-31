"""TD-MPC2 agent."""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from rl_hockey.common.agent import Agent
from rl_hockey.common.buffer import OpponentCloningBuffer, TDMPC2ReplayBuffer
from rl_hockey.TD_MPC2.model_dynamics_opponent import DynamicsOpponent
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
from rl_hockey.TD_MPC2.model_opponent_cloning import OpponentCloning
from rl_hockey.TD_MPC2.model_policy import Policy
from rl_hockey.TD_MPC2.model_q_ensemble import QEnsemble
from rl_hockey.TD_MPC2.model_reward import Reward
from rl_hockey.TD_MPC2.model_termination import Termination
from rl_hockey.TD_MPC2.mppi_planner_simple import MPPIPlannerSimplePaper
from rl_hockey.TD_MPC2.util import RunningScale, soft_ce, two_hot_inv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class DynamicsWithOpponentWrapper(torch.nn.Module):
    """Wraps DynamicsOpponent with opponent cloning; predicts opponent actions and samples opponent per forward."""

    def __init__(
        self, dynamics_opponent, opponent_cloning_networks, current_opponent_id=0
    ):
        super().__init__()
        self.dynamics_opponent = dynamics_opponent
        self.opponent_cloning_networks = opponent_cloning_networks
        self.current_opponent_id = current_opponent_id
        self.opponent_ids = (
            list(opponent_cloning_networks.keys()) if opponent_cloning_networks else []
        )
        self.force_opponent_id = None

    def forward(self, latent, action):
        if self.force_opponent_id is not None:
            opponent_id = self.force_opponent_id
        elif self.opponent_ids:
            opponent_id = self.opponent_ids[
                torch.randint(0, len(self.opponent_ids), (1,)).item()
            ]
        else:
            opponent_id = None

        if opponent_id is not None and opponent_id in self.opponent_cloning_networks:
            cloning_network = self.opponent_cloning_networks[opponent_id]["network"]
            with torch.no_grad():
                opponent_action = cloning_network.mean_action(latent)
        else:
            opponent_action = torch.zeros_like(action)
        return self.dynamics_opponent(latent, action, opponent_action)

    def set_current_opponent(self, opponent_id):
        """Force a specific opponent for non-planning rollouts."""
        self.force_opponent_id = opponent_id

    def clear_forced_opponent(self):
        self.force_opponent_id = None

    def update_opponent_networks(self, opponent_cloning_networks):
        """Refresh cloning networks after load."""
        self.opponent_cloning_networks = opponent_cloning_networks
        self.opponent_ids = (
            list(opponent_cloning_networks.keys()) if opponent_cloning_networks else []
        )


def _load_state_dict_compat(model, state_dict, strict=True):
    """Load state_dict; handle torch.compile _orig_mod key prefix."""
    if not state_dict:
        return None
    first_key = next(iter(state_dict.keys()))
    has_prefix = first_key.startswith("_orig_mod.")
    if hasattr(model, "_orig_mod"):
        if not has_prefix:
            return model._orig_mod.load_state_dict(state_dict, strict=strict)
    elif has_prefix:
        stripped = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        return model.load_state_dict(stripped, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


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
        use_amp=True,
        opponent_simulation_enabled=False,
        opponent_cloning_frequency=5000,
        opponent_cloning_steps=20,
        opponent_cloning_samples=512,
        opponent_agents=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.opponent_simulation_enabled = opponent_simulation_enabled
        self.opponent_cloning_frequency = opponent_cloning_frequency
        self.opponent_cloning_steps = opponent_cloning_steps
        self.opponent_cloning_samples = opponent_cloning_samples
        self.opponent_agents = opponent_agents if opponent_agents is not None else []
        self.training_step_counter = 0

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
        self.use_amp = use_amp and torch.cuda.is_available()

        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

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

        if self.opponent_simulation_enabled:
            self.dynamics = DynamicsOpponent(
                latent_dim,
                action_dim,
                action_opponent_dim=action_dim,
                hidden_dim=self.hidden_dim_dict["dynamics"],
                simnorm_temperature=simnorm_temperature,
            ).to(self.device)
        else:
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

        self.opponent_cloning_networks = {}
        self.opponent_cloning_buffers = {}
        self.loaded_opponent_agents = {}
        self.dynamics_wrapper = None
        if self.opponent_simulation_enabled:
            self._initialize_opponent_simulation()

        fused_available = "fused" in torch.optim.Adam.__init__.__code__.co_varnames
        optimizer_kwargs = {"lr": self.lr, "capturable": True}
        if fused_available and torch.cuda.is_available():
            optimizer_kwargs["fused"] = True

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
            **optimizer_kwargs,
        )
        pi_optimizer_kwargs = {"lr": self.lr, "eps": 1e-5, "capturable": True}
        if fused_available and torch.cuda.is_available():
            pi_optimizer_kwargs["fused"] = True
        self.pi_optimizer = torch.optim.Adam(
            list(self.policy.parameters()), **pi_optimizer_kwargs
        )

        self.scale = RunningScale(tau=self.tau).to(self.device)

        if self.opponent_simulation_enabled:
            if self.dynamics_wrapper is None:
                self.dynamics_wrapper = DynamicsWithOpponentWrapper(
                    self.dynamics, self.opponent_cloning_networks, current_opponent_id=0
                )
            else:
                self.dynamics_wrapper.update_opponent_networks(
                    self.opponent_cloning_networks
                )
            dynamics_for_planner = self.dynamics_wrapper
        else:
            dynamics_for_planner = self.dynamics

        self.planner = MPPIPlannerSimplePaper(
            dynamics_for_planner,
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
        buffer_device = self.config.get("buffer_device", "cpu")
        self.buffer = TDMPC2ReplayBuffer(
            max_size=capacity,
            horizon=self.horizon,
            batch_size=batch_size,
            use_torch_tensors=True,
            device=self.device,  # Where sampled batches go (for training)
            buffer_device=buffer_device,  # Where episode data is stored
            pin_memory=(
                buffer_device == "cpu"
            ),  # Pin memory for faster CPU->GPU transfer
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
                logger.debug("Successfully compiled encoder")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for encoder: {e}. Using eager mode."
                )

            try:
                self.dynamics = torch.compile(self.dynamics, mode="reduce-overhead")
                logger.debug("Successfully compiled dynamics")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for dynamics: {e}. Using eager mode."
                )

            try:
                self.reward = torch.compile(self.reward, mode="reduce-overhead")
                logger.debug("Successfully compiled reward")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for reward: {e}. Using eager mode."
                )

            try:
                self.termination = torch.compile(
                    self.termination, mode="reduce-overhead"
                )
                logger.debug("Successfully compiled termination")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for termination: {e}. Using eager mode."
                )

            try:
                self.q_ensemble = torch.compile(self.q_ensemble, mode="reduce-overhead")
                logger.debug("Successfully compiled q_ensemble")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for q_ensemble: {e}. Using eager mode."
                )

            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
                logger.debug("Successfully compiled policy")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for policy: {e}. Using eager mode."
                )

        self.planner.to(self.device)

    def store_transition(self, transition, winner=None):
        """Store transition in replay buffer. winner used for reward shaping."""
        self.buffer.store(transition, winner=winner)

    @torch.no_grad()
    def rollout_dynamics_multi_step(
        self, z0, action_sequence, max_horizon, opponent_id=None
    ):
        """Roll out dynamics from z0; returns {1: z_1, ..., max_horizon: z_h}. opponent_id forces opponent if set."""
        if opponent_id is not None and self.dynamics_wrapper is not None:
            self.dynamics_wrapper.set_current_opponent(opponent_id)
        elif self.dynamics_wrapper is not None:
            self.dynamics_wrapper.clear_forced_opponent()

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
            dynamics_model = (
                self.dynamics_wrapper
                if self.dynamics_wrapper is not None
                else self.dynamics
            )
            z = dynamics_model(z, a)

            out[h] = z.squeeze(0).cpu().numpy()
        return out

    @torch.no_grad()
    def act(self, obs, deterministic=False, t0=False, opponent_id=None):
        """Select action via MPC. opponent_id ignored; planning samples opponents per trajectory."""
        if self.dynamics_wrapper is not None:
            self.dynamics_wrapper.clear_forced_opponent()

        obs = torch.FloatTensor(obs).to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
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
        opponent_id=None,
    ):
        """Select action via MPC and return stats. opponent_id ignored."""
        if self.dynamics_wrapper is not None:
            self.dynamics_wrapper.clear_forced_opponent()

        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_batch = obs_tensor.unsqueeze(0)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
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

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            policy_action = self.policy.mean_action(z.unsqueeze(0)).squeeze(0)

            q_logits_list = self.q_ensemble(z.unsqueeze(0), action.unsqueeze(0))
            from rl_hockey.TD_MPC2.util import two_hot_inv

            q_values = []
            for q_logits in q_logits_list:
                q_val = two_hot_inv(
                    q_logits, self.num_bins, self.vmin, self.vmax
                ).item()
                q_values.append(q_val)

            stats["q_values"] = q_values
            stats["q_min"] = min(q_values)
            stats["q_max"] = max(q_values)
            stats["q_mean"] = sum(q_values) / len(q_values)
            q_std_val = (
                torch.tensor(q_values).std().item() if len(q_values) > 1 else 0.0
            )
            stats["q_std"] = q_std_val

            stats["q_spread"] = max(q_values) - min(q_values)
            q_mean_val = stats["q_mean"]
            stats["q_coefficient_of_variation"] = (
                q_std_val / abs(q_mean_val) if abs(q_mean_val) > 1e-8 else 0.0
            )

            q_logits_policy = self.q_ensemble(
                z.unsqueeze(0), policy_action.unsqueeze(0)
            )
            q_values_policy = []
            for q_logits in q_logits_policy:
                q_val = two_hot_inv(
                    q_logits, self.num_bins, self.vmin, self.vmax
                ).item()
                q_values_policy.append(q_val)

            stats["q_policy_values"] = q_values_policy
            stats["q_policy_min"] = min(q_values_policy)
            stats["q_policy_max"] = max(q_values_policy)
            stats["q_policy_mean"] = sum(q_values_policy) / len(q_values_policy)

            dynamics_model = (
                self.dynamics_wrapper
                if self.dynamics_wrapper is not None
                else self.dynamics
            )
            z_next_pred = dynamics_model(z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
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
                stats["dynamics_prediction_error_mse"] = (
                    (dynamics_error**2).mean().item()
                )
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
    def act_batch(self, obs_batch, deterministic=False, t0s=None, opponent_id=None):
        """Select actions for batch. opponent_id ignored."""
        if self.dynamics_wrapper is not None:
            self.dynamics_wrapper.clear_forced_opponent()

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        num_envs = obs_batch.shape[0]

        if self.planners is None or len(self.planners) != num_envs:
            self.planners = [copy.deepcopy(self.planner) for _ in range(num_envs)]

        with torch.amp.autocast("cuda", enabled=self.use_amp):
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
            with torch.amp.autocast("cuda", enabled=self.use_amp):
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

        cloning_losses = {}
        if self.opponent_simulation_enabled and self.opponent_cloning_frequency > 0:
            if (
                self.training_step_counter % self.opponent_cloning_frequency == 0
                and self.training_step_counter > 0
            ):
                logger.info(
                    f"Training opponent cloning networks at step {self.training_step_counter}"
                )
                cloning_losses = self._train_opponent_cloning()
                all_losses.update(cloning_losses)

        for _ in range(steps):
            self.training_step_counter += 1
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

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
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        z_flat = self.encoder(obs_flat)
                    z_seq = z_flat.reshape(
                        batch_size_actual, horizon + 1, self.latent_dim
                    )

                    with torch.no_grad():
                        next_z = z_seq[:, 1:].detach()
                        td_targets = self._compute_td_targets(
                            next_z, rewards_seq, dones_seq, horizon
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

                    lambda_weights = self.lambda_coef ** torch.arange(
                        horizon, device=self.device, dtype=torch.float32
                    )

                    for t in range(horizon):
                        a_t = actions_seq[:, t]
                        z_target = z_seq[:, t + 1].detach()

                        with torch.amp.autocast("cuda", enabled=self.use_amp):
                            if self.opponent_simulation_enabled:
                                if self.opponent_cloning_networks:
                                    opponent_id = 0
                                    if opponent_id in self.opponent_cloning_networks:
                                        cloning_network = (
                                            self.opponent_cloning_networks[opponent_id][
                                                "network"
                                            ]
                                        )
                                        with torch.no_grad():
                                            opponent_action_t = (
                                                cloning_network.mean_action(z_pred)
                                            )
                                    else:
                                        opponent_action_t = torch.zeros_like(a_t)
                                else:
                                    opponent_action_t = torch.zeros_like(a_t)
                                z_next_pred = self.dynamics(
                                    z_pred, a_t, opponent_action_t
                                )
                            else:
                                z_next_pred = self.dynamics(z_pred, a_t)
                        zs[t + 1] = z_next_pred

                        consistency_loss = consistency_loss + lambda_weights[
                            t
                        ] * F.mse_loss(z_next_pred, z_target)

                        z_pred = z_next_pred

                    consistency_loss = consistency_loss / horizon

                    _zs = zs[:-1]
                    _zs_flat = _zs.reshape(-1, self.latent_dim)
                    _actions_flat = actions_seq.permute(1, 0, 2).reshape(
                        -1, self.action_dim
                    )

                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        reward_preds = self.reward(_zs_flat, _actions_flat)
                        q_preds_all = self.q_ensemble(_zs_flat, _actions_flat)

                    rewards_flat = rewards_seq.permute(1, 0, 2).reshape(-1, 1)
                    reward_loss_per_sample = soft_ce(
                        reward_preds, rewards_flat, self.num_bins, self.vmin, self.vmax
                    )
                    reward_loss_per_step = reward_loss_per_sample.reshape(
                        horizon, batch_size_actual
                    ).mean(dim=1)
                    reward_loss = (
                        lambda_weights * reward_loss_per_step
                    ).sum() / horizon

                    td_targets_flat = td_targets.permute(1, 0, 2).reshape(-1, 1)
                    value_loss = 0.0
                    for q_pred in q_preds_all:
                        value_loss_per_sample = soft_ce(
                            q_pred, td_targets_flat, self.num_bins, self.vmin, self.vmax
                        )
                        value_loss_per_step = value_loss_per_sample.reshape(
                            horizon, batch_size_actual
                        ).mean(dim=1)
                        value_loss = (
                            value_loss + (lambda_weights * value_loss_per_step).sum()
                        )
                    value_loss = value_loss / (horizon * self.num_q)
                    zs_bh = zs[1:].permute(1, 0, 2).reshape(-1, self.latent_dim)
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
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

            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm_encoder = self._grad_norm(self.encoder.parameters())
                grad_norm_dynamics = self._grad_norm(self.dynamics.parameters())
                grad_norm_reward = self._grad_norm(self.reward.parameters())
                grad_norm_termination = self._grad_norm(self.termination.parameters())
                grad_norm_q_ensemble = self._grad_norm(self.q_ensemble.parameters())
                torch.nn.utils.clip_grad_norm_(
                    self._model_params, max_norm=self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm_encoder = self._grad_norm(self.encoder.parameters())
                grad_norm_dynamics = self._grad_norm(self.dynamics.parameters())
                grad_norm_reward = self._grad_norm(self.reward.parameters())
                grad_norm_termination = self._grad_norm(self.termination.parameters())
                grad_norm_q_ensemble = self._grad_norm(self.q_ensemble.parameters())
                torch.nn.utils.clip_grad_norm_(
                    self._model_params, max_norm=self.grad_clip_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
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

    @torch.no_grad()
    def _compute_td_targets(self, next_z, rewards_seq, dones_seq, horizon):
        """Compute TD targets for all timesteps; batched."""
        batch_size = next_z.shape[0]
        if self.n_step == 1:
            next_z_flat = next_z.reshape(-1, self.latent_dim)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                next_actions, _, _, _ = self.policy.sample(next_z_flat)
                target_q_flat = self.target_q_ensemble.min(next_z_flat, next_actions)
            target_q = target_q_flat.reshape(batch_size, horizon, 1)
            td_targets = rewards_seq + self.gamma * (1.0 - dones_seq) * target_q
        else:
            td_targets = torch.empty(
                batch_size, horizon, 1, device=self.device, dtype=rewards_seq.dtype
            )
            for t in range(horizon):
                n = min(self.n_step, horizon - t)
                gamma_powers = (
                    self.gamma
                    ** torch.arange(n, device=self.device, dtype=rewards_seq.dtype)
                ).view(1, n, 1)
                reward_sum = (rewards_seq[:, t : t + n] * gamma_powers).sum(dim=1)

                z_bootstrap = next_z[:, min(t + n - 1, horizon - 1)]
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    next_action, _, _, _ = self.policy.sample(z_bootstrap)
                    target_q = self.target_q_ensemble.min(z_bootstrap, next_action)

                d_n = dones_seq[:, min(t + n - 1, horizon - 1)]
                bootstrap = (self.gamma**n) * (1.0 - d_n) * target_q
                td_targets[:, t] = reward_sum + bootstrap

        return td_targets

    def _init_detached_q_ensemble(self):
        """No-op; detachment done in _update_policy via functional_call."""
        return

    def _grad_norm(self, parameters):
        """Compute gradient norm without clipping (faster than clip_grad_norm_ with inf)."""
        parameters = list(parameters)
        if len(parameters) == 0:
            return torch.tensor(0.0, device=self.device)
        grads = [p.grad for p in parameters if p.grad is not None]
        if len(grads) == 0:
            return torch.tensor(0.0, device=self.device)
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
        )
        return total_norm

    def _q_ensemble_detached(self, latent, action):
        """Forward through Q-ensemble with detached parameters."""
        params = {
            name: param.detach() for name, param in self.q_ensemble.named_parameters()
        }
        buffers = dict(self.q_ensemble.named_buffers())
        return functional_call(self.q_ensemble, {**params, **buffers}, (latent, action))

    def _update_policy(self, zs):
        """Update policy to maximize Q + entropy."""
        self.pi_optimizer.zero_grad()
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        num_states = zs.shape[0]
        batch_size = zs.shape[1]
        zs_flat = zs.reshape(-1, zs.shape[-1])

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            actions, log_probs, _, scaled_entropies = self.policy.sample(zs_flat)
            q_logits_all = self._q_ensemble_detached(zs_flat, actions)

            q_logits_stacked = torch.stack(q_logits_all, dim=0)
            q_values = two_hot_inv(
                q_logits_stacked, self.num_bins, self.vmin, self.vmax
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
                -(self.entropy_coef * scaled_entropies + qs_scaled).mean(dim=(1, 2))
                * rho
            ).mean()

        if self.scaler is not None:
            self.scaler.scale(pi_loss).backward()
            self.scaler.unscale_(self.pi_optimizer)
            grad_norm_policy = self._grad_norm(self.policy.parameters())
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.grad_clip_norm
            )
            self.scaler.step(self.pi_optimizer)
            self.scaler.update()
        else:
            pi_loss.backward()

            # Compute gradient norm using helper, then clip
            grad_norm_policy = self._grad_norm(self.policy.parameters())
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.grad_clip_norm
            )
            self.pi_optimizer.step()

        self.pi_optimizer.zero_grad(set_to_none=True)
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
        checkpoint = {
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
            "opponent_simulation_enabled": self.opponent_simulation_enabled,
            "training_step_counter": self.training_step_counter,
        }
        if self.opponent_simulation_enabled:
            checkpoint["opponent_cloning_frequency"] = self.opponent_cloning_frequency
            checkpoint["opponent_cloning_steps"] = self.opponent_cloning_steps
            checkpoint["opponent_cloning_samples"] = self.opponent_cloning_samples
            checkpoint["opponent_agents"] = self.opponent_agents
            opponent_cloning_states = {}
            for opponent_id, cloning_info in self.opponent_cloning_networks.items():
                opponent_cloning_states[opponent_id] = {
                    "network": cloning_info["network"].state_dict(),
                    "optimizer": cloning_info["optimizer"].state_dict(),
                }
            checkpoint["opponent_cloning_networks"] = opponent_cloning_states

        torch.save(checkpoint, path)

    def load(self, path):
        """Load agent"""
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_opponent_simulation = checkpoint.get(
            "opponent_simulation_enabled", False
        )
        if checkpoint_opponent_simulation and not self.opponent_simulation_enabled:
            logger.warning(
                "Checkpoint was trained with opponent simulation, but current agent has it disabled. "
                "Opponent cloning networks will not be loaded."
            )
        elif not checkpoint_opponent_simulation and self.opponent_simulation_enabled:
            logger.warning(
                "Checkpoint was trained without opponent simulation, but current agent has it enabled. "
                "Opponent cloning networks will be initialized from scratch."
            )

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

        _load_state_dict_compat(self.encoder, checkpoint["encoder"])
        _load_state_dict_compat(self.dynamics, checkpoint["dynamics"])

        if checkpoint_num_bins is not None and checkpoint_num_bins != self.num_bins:
            load_result = _load_state_dict_compat(
                self.reward, checkpoint["reward"], strict=False
            )
            missing_keys, unexpected_keys = (
                load_result.missing_keys,
                load_result.unexpected_keys,
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
            _load_state_dict_compat(self.reward, checkpoint["reward"])

        if "termination" in checkpoint:
            _load_state_dict_compat(self.termination, checkpoint["termination"])
        else:
            logger.warning(
                "Termination model not found in checkpoint, initializing from scratch."
            )

        if checkpoint_num_bins is not None and checkpoint_num_bins != self.num_bins:
            load_result = _load_state_dict_compat(
                self.q_ensemble, checkpoint["q_ensemble"], strict=False
            )
            missing_keys, unexpected_keys = (
                load_result.missing_keys,
                load_result.unexpected_keys,
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
            _load_state_dict_compat(self.q_ensemble, checkpoint["q_ensemble"])

        _load_state_dict_compat(self.policy, checkpoint["policy"])

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
            load_result = _load_state_dict_compat(
                self.target_q_ensemble, checkpoint["q_ensemble"], strict=False
            )
            missing_keys, unexpected_keys = (
                load_result.missing_keys,
                load_result.unexpected_keys,
            )
            if missing_keys:
                logger.debug(
                    f"Target Q ensemble missing keys (expected): {missing_keys}"
                )
            if unexpected_keys:
                logger.debug(f"Target Q ensemble unexpected keys: {unexpected_keys}")
        else:
            _load_state_dict_compat(self.target_q_ensemble, checkpoint["q_ensemble"])

        self._init_detached_q_ensemble()
        if "training_step_counter" in checkpoint:
            self.training_step_counter = checkpoint["training_step_counter"]
        if checkpoint_opponent_simulation and self.opponent_simulation_enabled:
            if "opponent_cloning_networks" in checkpoint:
                opponent_cloning_states = checkpoint["opponent_cloning_networks"]
                hidden_dim = self.hidden_dim_dict.get("policy", [256, 256, 256])
                fused_available = (
                    "fused" in torch.optim.Adam.__init__.__code__.co_varnames
                )
                optimizer_kwargs = {"lr": self.lr, "eps": 1e-5, "capturable": True}
                if fused_available and torch.cuda.is_available():
                    optimizer_kwargs["fused"] = True
                for opponent_id, states in opponent_cloning_states.items():
                    if opponent_id not in self.opponent_cloning_networks:
                        cloning_network = OpponentCloning(
                            latent_dim=self.latent_dim,
                            action_dim=self.action_dim,
                            hidden_dim=hidden_dim,
                            log_std_min=self.log_std_min,
                            log_std_max=self.log_std_max,
                        ).to(self.device)
                        cloning_optimizer = torch.optim.Adam(
                            cloning_network.parameters(), **optimizer_kwargs
                        )
                        self.opponent_cloning_networks[opponent_id] = {
                            "network": cloning_network,
                            "optimizer": cloning_optimizer,
                        }
                    try:
                        _load_state_dict_compat(
                            self.opponent_cloning_networks[opponent_id]["network"],
                            states["network"],
                        )
                        self.opponent_cloning_networks[opponent_id][
                            "optimizer"
                        ].load_state_dict(states["optimizer"])
                        logger.info(f"Loaded opponent cloning network {opponent_id}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load opponent cloning network {opponent_id}: {e}"
                        )
            if self.dynamics_wrapper is not None:
                self.dynamics_wrapper.update_opponent_networks(
                    self.opponent_cloning_networks
                )
            else:
                self.dynamics_wrapper = DynamicsWithOpponentWrapper(
                    self.dynamics, self.opponent_cloning_networks, current_opponent_id=0
                )
                self.planner.dynamics = self.dynamics_wrapper

    def set_current_opponent(self, opponent_id):
        """Force opponent for rollouts; planning still samples randomly."""
        if self.dynamics_wrapper is not None:
            self.dynamics_wrapper.set_current_opponent(opponent_id)
            logger.info(
                f"Forced opponent to {opponent_id} (note: planning uses random sampling)"
            )
        else:
            logger.warning(
                "Opponent simulation not enabled, cannot set current opponent"
            )

    def store_opponent_action(self, obs, opponent_action, opponent_id):
        """Store opponent demonstration (obs, action) pair in the corresponding buffer.

        Use this only when the acting opponent is one of your loaded reference bots.
        Prefer collect_opponent_demonstrations(obs) to fill all buffers in parallel
        regardless of who the training opponent is.

        Args:
            obs: Observation (numpy array or torch tensor)
            opponent_action: Action taken by opponent (numpy array or torch tensor)
            opponent_id: ID of the opponent (int)
        """
        if not self.opponent_simulation_enabled:
            return

        if opponent_id not in self.opponent_cloning_buffers:
            logger.warning(
                f"Opponent {opponent_id} has no cloning buffer, cannot store action"
            )
            return

        self.opponent_cloning_buffers[opponent_id].add(obs, opponent_action)

    def collect_opponent_demonstrations(self, obs_agent2):
        """Run each loaded reference opponent on the given observation and store
        (obs, action) in that opponent's cloning buffer.

        Call this every environment step with the opponent's observation (obs_agent2).
        Buffers fill regardless of who the actual training opponent is; we simulate
        what each reference opponent would do at this step in parallel.

        Args:
            obs_agent2: Observation from opponent's perspective (numpy or torch),
                shape (obs_dim,) or (1, obs_dim).
        """
        if not self.opponent_simulation_enabled:
            return

        if not self.loaded_opponent_agents or not self.opponent_cloning_buffers:
            return

        obs = obs_agent2
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
            obs_flat = obs_np.reshape(-1) if obs_np.size > 0 else obs_np
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
            obs_flat = obs_np.reshape(-1) if obs_np.size > 0 else obs_np

        for opponent_id in list(self.opponent_cloning_buffers.keys()):
            if opponent_id not in self.loaded_opponent_agents:
                continue

            opponent_agent = self.loaded_opponent_agents[opponent_id]["agent"]

            with torch.inference_mode():
                if hasattr(opponent_agent, "policy") and hasattr(
                    opponent_agent, "encoder"
                ):
                    obs_batch = (
                        torch.from_numpy(obs_np).float().reshape(1, -1).to(self.device)
                    )
                    with torch.amp.autocast("cuda", enabled=False):
                        z = opponent_agent.encoder(obs_batch)
                        action = opponent_agent.policy.mean_action(z)
                    action = action[0].cpu().numpy()
                elif hasattr(opponent_agent, "act_batch"):
                    batch = obs_flat.reshape(1, -1)
                    action_batch = opponent_agent.act_batch(batch, deterministic=True)
                    action = np.asarray(action_batch[0], dtype=np.float32)
                elif hasattr(opponent_agent, "act"):
                    action = np.asarray(
                        opponent_agent.act(obs_flat, deterministic=True),
                        dtype=np.float32,
                    )
                else:
                    continue

                self.opponent_cloning_buffers[opponent_id].add(obs_flat, action)

    def _train_opponent_cloning(self):
        """Train cloning networks using stored opponent demonstrations.

        Uses OpponentCloningBuffer for each opponent which stores (obs, action) pairs
        collected during episode rollouts. Samples from buffer, encodes once, then
        runs multiple gradient steps on minibatches. No opponent forward passes needed.
        """
        if not self.opponent_simulation_enabled:
            logger.warning("Opponent simulation not enabled, skipping cloning training")
            return {}

        if not self.opponent_cloning_networks:
            logger.warning(
                "No opponent cloning networks available, skipping cloning training"
            )
            return {}

        if not self.opponent_cloning_buffers:
            logger.warning(
                "No opponent cloning buffers available, skipping cloning training"
            )
            return {}

        encoder_requires_grad = [p.requires_grad for p in self.encoder.parameters()]
        for p in self.encoder.parameters():
            p.requires_grad = False

        cloning_losses = {}
        minibatch_size = 256

        logger.info(
            f"Training {len(self.opponent_cloning_networks)} opponent cloning network(s)"
        )

        for opponent_id, cloning_info in self.opponent_cloning_networks.items():
            if opponent_id not in self.opponent_cloning_buffers:
                logger.warning(
                    f"Opponent {opponent_id} has no cloning buffer, skipping"
                )
                continue

            cloning_buffer = self.opponent_cloning_buffers[opponent_id]
            if len(cloning_buffer) < self.opponent_cloning_samples:
                logger.warning(
                    f"Opponent {opponent_id} buffer has insufficient samples "
                    f"({len(cloning_buffer)} < {self.opponent_cloning_samples}), skipping"
                )
                continue

            cloning_network = cloning_info["network"]
            cloning_optimizer = cloning_info["optimizer"]

            try:
                obs_all, action_all = cloning_buffer.sample(
                    self.opponent_cloning_samples
                )
            except RuntimeError as e:
                logger.warning(
                    f"Failed to sample from opponent {opponent_id} buffer: {e}"
                )
                continue

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    latent_all = self.encoder(obs_all)
            latent_all = latent_all.float()

            total_loss = 0.0
            n_steps = 0
            num_samples = latent_all.shape[0]
            actual_minibatch_size = min(minibatch_size, num_samples)

            for step_idx in range(self.opponent_cloning_steps):
                perm = torch.randperm(num_samples, device=self.device)[
                    :actual_minibatch_size
                ]
                latent_mb = latent_all[perm]
                target_mb = action_all[perm]

                with torch.amp.autocast("cuda", enabled=False):
                    cloned_actions = cloning_network.mean_action(latent_mb)
                loss = F.mse_loss(cloned_actions, target_mb.to(cloned_actions.dtype))
                cloning_optimizer.zero_grad()
                loss.backward()
                cloning_optimizer.step()
                total_loss += loss.item()
                n_steps += 1

            avg_loss = total_loss / n_steps if n_steps > 0 else 0.0
            cloning_losses[f"opponent_{opponent_id}_cloning_loss"] = avg_loss
            logger.info(
                f"  -> Opponent {opponent_id}: loss={avg_loss:.6f}, "
                f"buffer_size={len(cloning_buffer)}, steps={n_steps}"
            )

        for p, requires_grad in zip(self.encoder.parameters(), encoder_requires_grad):
            p.requires_grad = requires_grad

        if not cloning_losses:
            logger.warning("No opponent cloning losses computed - check buffer filling")

        return cloning_losses

    def _initialize_opponent_simulation(self):
        """Load opponent agents and create cloning networks."""
        if not self.opponent_agents:
            logger.info(
                "Opponent simulation enabled but no opponent agents specified; "
                "cloning networks will be created from checkpoint when loading."
            )
            return

        for i, opponent_info in enumerate(self.opponent_agents):
            opponent_type = opponent_info.get("type")
            opponent_path = opponent_info.get("path")

            if not opponent_type or not opponent_path:
                logger.warning(f"Opponent {i} missing type or path, skipping")
                continue

            try:
                opponent_agent = self._load_opponent_agent(opponent_type, opponent_path)
                self.loaded_opponent_agents[i] = {
                    "type": opponent_type,
                    "agent": opponent_agent,
                    "path": opponent_path,
                }
                cloning_network = OpponentCloning(
                    latent_dim=self.latent_dim,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim_dict.get("policy", [256, 256, 256]),
                    log_std_min=self.log_std_min,
                    log_std_max=self.log_std_max,
                ).to(self.device)
                init_policy(cloning_network)
                fused_available = (
                    "fused" in torch.optim.Adam.__init__.__code__.co_varnames
                )
                optimizer_kwargs = {"lr": self.lr, "eps": 1e-5, "capturable": True}
                if fused_available and torch.cuda.is_available():
                    optimizer_kwargs["fused"] = True

                cloning_optimizer = torch.optim.Adam(
                    cloning_network.parameters(), **optimizer_kwargs
                )

                self.opponent_cloning_networks[i] = {
                    "network": cloning_network,
                    "optimizer": cloning_optimizer,
                }

                cloning_buffer = OpponentCloningBuffer(
                    max_size=50000,
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    use_torch_tensors=True,
                    device=self.device,
                )
                self.opponent_cloning_buffers[i] = cloning_buffer

                logger.info(
                    f"Successfully initialized opponent {i}: type={opponent_type}, path={opponent_path}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize opponent {i} (type={opponent_type}, path={opponent_path}): {e}"
                )
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

        logger.info("Opponent simulation initialization complete:")
        logger.info(
            f"  - Cloning networks created: {len(self.opponent_cloning_networks)}"
        )
        logger.info(
            f"  - Cloning buffers created: {len(self.opponent_cloning_buffers)}"
        )
        logger.info(f"  - Opponent agents loaded: {len(self.loaded_opponent_agents)}")
        if len(self.loaded_opponent_agents) == 0:
            logger.warning("  No opponents successfully loaded.")

    def _load_opponent_agent(self, opponent_type, opponent_path):
        """Load opponent from disk; for TDMPC2, match checkpoint hyperparameters."""
        if opponent_type.upper() == "TDMPC2":
            import torch

            from rl_hockey.TD_MPC2.tdmpc2 import TDMPC2

            checkpoint = torch.load(
                opponent_path, map_location=self.device, weights_only=False
            )
            config = checkpoint.get("config", {})
            latent_dim = checkpoint.get("latent_dim") or config.get("latent_dim", 512)
            hidden_dim = checkpoint.get("hidden_dim") or config.get("hidden_dim")
            num_q = checkpoint.get("num_q") or config.get("num_q", 5)
            horizon = checkpoint.get("horizon") or config.get("horizon", 5)
            gamma = checkpoint.get("gamma") or config.get("gamma", 0.99)

            logger.info(
                f"Creating opponent TDMPC2 with latent_dim={latent_dim}, hidden_dim={hidden_dim}"
            )
            agent = TDMPC2(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_q=num_q,
                horizon=horizon,
                gamma=gamma,
                device=self.device,
                opponent_simulation_enabled=False,
            )
            agent.load(opponent_path)
            if hasattr(agent, "eval"):
                agent.eval()
            elif hasattr(agent, "encoder"):
                agent.encoder.eval()
                if hasattr(agent, "dynamics"):
                    agent.dynamics.eval()
                if hasattr(agent, "policy"):
                    agent.policy.eval()
            return agent
        elif opponent_type.upper() == "SAC":
            from rl_hockey.sac.sac import SAC

            agent = SAC(
                state_dim=self.obs_dim,
                action_dim=self.action_dim,
            )
            agent.load(opponent_path)
            if hasattr(agent, "eval"):
                agent.eval()
            elif hasattr(agent, "actor"):
                agent.actor.eval()
            return agent
        elif opponent_type.upper() == "TD3":
            from rl_hockey.td3.td3 import TD3

            agent = TD3(
                state_dim=self.obs_dim,
                action_dim=self.action_dim,
            )
            agent.load(opponent_path)
            if hasattr(agent, "eval"):
                agent.eval()
            elif hasattr(agent, "actor"):
                agent.actor.eval()
            return agent
        elif opponent_type.upper() == "DECOYPOLICY":
            from rl_hockey.Decoy_Policy.decoy_policy import DecoyPolicy

            agent = DecoyPolicy(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
            )
            agent.load(opponent_path)
            if hasattr(agent, "network"):
                agent.network.eval()
            return agent
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

    def log_architecture(self):
        """Log agent architecture and config."""

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines = []
        lines.append("=" * 80)
        lines.append("TD-MPC2 agent architecture")
        lines.append("=" * 80)
        lines.append(f"Observation dim: {self.obs_dim}")
        lines.append(f"Action dim: {self.action_dim}")
        lines.append(f"Latent dim: {self.latent_dim}")
        lines.append("Hidden dims (per network):")
        for network_type, hidden_dims in self.hidden_dim_dict.items():
            lines.append(f"  {network_type}: {hidden_dims}")
        lines.append(f"Device: {self.device}")
        lines.append("")

        lines.append("Configuration:")
        for key, value in self.config.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        lines.append("Model-based components:")
        lines.append("")
        lines.append("1. Encoder:")
        lines.append(str(self.encoder))
        lines.append(f"   trainable parameters: {count_parameters(self.encoder):,}")
        lines.append("")
        lines.append("2. Dynamics:")
        lines.append(str(self.dynamics))
        lines.append(f"   trainable parameters: {count_parameters(self.dynamics):,}")
        lines.append("")
        lines.append("3. Reward:")
        lines.append(str(self.reward))
        lines.append(f"   trainable parameters: {count_parameters(self.reward):,}")
        lines.append("")
        lines.append("3a. Termination:")
        lines.append(str(self.termination))
        lines.append(f"   trainable parameters: {count_parameters(self.termination):,}")
        lines.append("")
        lines.append(f"4. Q ensemble ({self.num_q} heads):")
        lines.append(str(self.q_ensemble))
        lines.append(f"   trainable parameters: {count_parameters(self.q_ensemble):,}")
        lines.append("")
        lines.append("5. Target Q ensemble:")
        lines.append(
            f"   trainable parameters: {count_parameters(self.target_q_ensemble):,}"
        )
        lines.append("")
        lines.append("6. Policy:")
        lines.append(str(self.policy))
        lines.append(f"   trainable parameters: {count_parameters(self.policy):,}")
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

        lines.append("Parameter summary:")
        lines.append(f"  World model: {model_params:,}")
        lines.append(f"  Policy Network: {count_parameters(self.policy):,}")
        lines.append(
            f"  Target Q Network: {count_parameters(self.target_q_ensemble):,}"
        )
        lines.append(f"  TOTAL TRAINABLE PARAMETERS: {total_params:,}")
        lines.append("")

        lines.append("Optimizers:")
        lines.append(
            f"  World Model + Q Optimizer: {self.optimizer.__class__.__name__} (LR: {self.lr})"
        )
        lines.append(
            f"  Policy Optimizer: {self.pi_optimizer.__class__.__name__} (LR: {self.lr})"
        )
        lines.append("")

        lines.append("MPPI planner:")
        lines.append(f"  Horizon: {self.horizon}")
        lines.append(f"  Samples per iteration: {self.num_samples}")
        lines.append(f"  Planning iterations: {self.num_iterations}")
        lines.append(f"  Temperature: {self.temperature}")
        if hasattr(self, "fast_mode"):
            lines.append(f"  Fast Mode: {self.fast_mode} (use policy network only)")
        lines.append("")

        if self.opponent_simulation_enabled:
            lines.append("Opponent simulation:")
            lines.append("  Enabled: True")
            lines.append(
                f"  Cloning Frequency: {self.opponent_cloning_frequency} training steps"
            )
            lines.append(
                f"  Cloning Training Steps: {self.opponent_cloning_steps} gradient steps per update"
            )
            lines.append(
                f"  Cloning Samples: {self.opponent_cloning_samples} observations per batch"
            )
            lines.append(f"  Number of Opponents: {len(self.opponent_agents)}")

            if self.opponent_agents:
                lines.append("  Opponent Agents:")
                for i, opp_info in enumerate(self.opponent_agents):
                    opp_type = opp_info.get("type", "Unknown")
                    opp_path = opp_info.get("path", "No path")
                    lines.append(f"    [{i}] Type: {opp_type}")
                    lines.append(f"        Path: {opp_path}")
            if self.opponent_cloning_networks:
                total_cloning_params = sum(
                    count_parameters(info["network"])
                    for info in self.opponent_cloning_networks.values()
                )
                lines.append(
                    f"  Total Cloning Network Parameters: {total_cloning_params:,}"
                )

            lines.append("  (MPPI samples opponent per trajectory)")
            lines.append("")
        else:
            lines.append("Opponent simulation: disabled")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
