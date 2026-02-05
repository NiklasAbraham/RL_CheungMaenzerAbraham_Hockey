"""Main TDMPC2 agent class."""

import copy
import logging

import torch

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from rl_hockey.common.agent import Agent
from rl_hockey.common.buffer import TDMPC2ReplayBuffer
from rl_hockey.TD_MPC2.core import inference, training
from rl_hockey.TD_MPC2.model_definition import (
    DynamicsOpponent,
    DynamicsSimple,
    DynamicsWithOpponentWrapper,
    Encoder,
    MPPIPlannerSimplePaper,
    Policy,
    QEnsemble,
    Reward,
    Termination,
    init_dynamics,
    init_encoder,
    init_policy,
    init_q_ensemble,
    init_reward,
    init_termination,
)
from rl_hockey.TD_MPC2.opponent import (
    collect_opponent_demonstrations,
    initialize_opponent_simulation,
    set_current_opponent,
    store_opponent_action,
)
from rl_hockey.TD_MPC2.persistence import checkpoint
from rl_hockey.TD_MPC2.planning import rollout_dynamics_multi_step
from rl_hockey.TD_MPC2.utils import RunningScale, log_architecture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class TDMPC2(Agent):
    """TD-MPC2 agent for model-based reinforcement learning."""

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
            "win_reward_bonus": win_reward_bonus,
            "win_reward_discount": win_reward_discount,
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

        self.policy = Policy(
            latent_dim,
            action_dim,
            self.hidden_dim_dict["policy"],
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        # Initialize all networks BEFORE creating target_q_ensemble
        # This ensures target_q_ensemble has the same initialization
        init_encoder(self.encoder)
        init_dynamics(self.dynamics)
        init_reward(self.reward)
        init_termination(self.termination)
        init_q_ensemble(self.q_ensemble)
        init_policy(self.policy)

        # Create target network AFTER initialization so it has matching weights
        self.target_q_ensemble = copy.deepcopy(self.q_ensemble)
        for param in self.target_q_ensemble.parameters():
            param.requires_grad = False

        self.opponent_cloning_networks = {}
        self.opponent_cloning_buffers = {}
        self.loaded_opponent_agents = {}
        self.dynamics_wrapper = None
        if self.opponent_simulation_enabled:
            initialize_opponent_simulation(self)

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

    def store_transition(self, transition, winner=None, env_id=None):
        """Store transition in replay buffer.
        
        Args:
            transition: (state, action, reward, next_state, done) tuple
            winner: Optional winner info for reward shaping (1=win, -1=loss, 0=draw)
            env_id: Optional environment ID for vectorized training. When provided,
                transitions are tracked per-environment to properly separate episodes.
        """
        self.buffer.store(transition, winner=winner, env_id=env_id)

    def rollout_dynamics_multi_step(
        self, z0, action_sequence, max_horizon, opponent_id=None
    ):
        """Roll out dynamics from z0; returns {1: z_1, ..., max_horizon: z_h}. opponent_id forces opponent if set."""
        return rollout_dynamics_multi_step(self, z0, action_sequence, max_horizon, opponent_id)

    def act(self, obs, deterministic=False, t0=False, opponent_id=None):
        """Select action via MPC. opponent_id ignored; planning samples opponents per trajectory."""
        return inference.act(self, obs, deterministic, t0, opponent_id)

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
        return inference.act_with_stats(
            self,
            obs,
            deterministic,
            prev_action,
            prev_latent,
            prev_predicted_next_latent,
            t0,
            opponent_id,
        )

    def act_batch(self, obs_batch, deterministic=False, t0s=None, opponent_id=None):
        """Select actions for batch. opponent_id ignored."""
        return inference.act_batch(self, obs_batch, deterministic, t0s, opponent_id)

    def evaluate(self, obs):
        """Evaluate state value using Q-ensemble."""
        return inference.evaluate(self, obs)

    def train(self, steps=1):
        """Train the agent for specified number of steps."""
        return training.train(self, steps)

    def save(self, path):
        """Save agent checkpoint."""
        checkpoint.save(self, path)

    def load(self, path):
        """Load agent checkpoint."""
        checkpoint.load(self, path)

    def set_current_opponent(self, opponent_id):
        """Force opponent for rollouts; planning still samples randomly."""
        set_current_opponent(self, opponent_id)

    def store_opponent_action(self, obs, opponent_action, opponent_id):
        """Store opponent demonstration (obs, action) pair in the corresponding buffer."""
        store_opponent_action(self, obs, opponent_action, opponent_id)

    def collect_opponent_demonstrations(self, obs_agent2):
        """Run each loaded reference opponent on the given observation and store (obs, action) in that opponent's cloning buffer."""
        collect_opponent_demonstrations(self, obs_agent2)

    def log_architecture(self):
        """Log agent architecture and config."""
        return log_architecture(self)
