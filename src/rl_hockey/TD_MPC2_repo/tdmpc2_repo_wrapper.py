"""
Wrapper for TD_MPC2_repo's TDMPC2 to integrate with the common Agent interface.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add parent directory to path to import rl_hockey modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from rl_hockey.common.agent import Agent
from rl_hockey.common.buffer import TDMPC2ReplayBuffer

# Import TD_MPC2_repo modules
# Add tdmpc2 directory to sys.path so relative imports work
repo_path = current_dir / "tdmpc2"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

try:
    from common.parser import cfg_to_dataclass, parse_cfg
    from common.seed import set_seed
    from omegaconf import OmegaConf
    from tdmpc2 import TDMPC2 as TDMPC2Repo
except ImportError as e:
    # If imports fail, provide helpful error message
    raise ImportError(
        f"Failed to import TD_MPC2_repo modules. "
        f"Make sure the TD_MPC2_repo/tdmpc2 directory exists and contains the required modules. "
        f"Error: {e}"
    )


class TDMPC2RepoWrapper(Agent):
    """
    Wrapper for TD_MPC2_repo's TDMPC2 that implements the common Agent interface.

    This wrapper allows using the reference TD-MPC2 implementation with the
    existing training infrastructure.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 512,
        hidden_dim: Optional[dict] = None,
        num_q: int = 5,
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        gamma: float = 0.99,
        horizon: int = 5,
        num_samples: int = 512,
        num_iterations: int = 6,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        capacity: int = 1000000,
        temperature: float = 0.5,
        batch_size: int = 256,
        device: str = "cuda",
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        tau: float = 0.01,
        grad_clip_norm: float = 20.0,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        termination_coef: float = 1.0,
        rho: float = 0.5,
        entropy_coef: float = 1e-4,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        min_std: float = 0.05,
        max_std: float = 2.0,
        discount_denom: float = 5.0,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        episodic: bool = False,
        mpc: bool = True,
        compile: bool = True,
        episode_length: int = 500,
        seed: int = 1,
        **kwargs,
    ):
        """
        Initialize TDMPC2RepoWrapper.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            latent_dim: Latent dimension for world model
            hidden_dim: Optional dict with network-specific hidden dimensions
            num_q: Number of Q-networks in ensemble
            lr: Learning rate
            enc_lr_scale: Encoder learning rate scale
            gamma: Discount factor
            horizon: Planning horizon
            num_samples: Number of samples for MPPI
            num_iterations: Number of MPPI iterations
            num_elites: Number of elite samples in MPPI
            num_pi_trajs: Number of policy trajectories in MPPI
            capacity: Replay buffer capacity
            temperature: MPPI temperature
            batch_size: Training batch size
            device: Device to use ('cuda' or 'cpu')
            num_bins: Number of bins for discrete regression
            vmin: Minimum value for value function
            vmax: Maximum value for value function
            tau: Soft update coefficient for target networks
            grad_clip_norm: Gradient clipping norm
            consistency_coef: Consistency loss coefficient
            reward_coef: Reward prediction loss coefficient
            value_coef: Value loss coefficient
            termination_coef: Termination prediction loss coefficient
            rho: Discount factor for multi-step losses
            entropy_coef: Entropy coefficient for policy
            log_std_min: Minimum log std for policy
            log_std_max: Maximum log std for policy
            min_std: Minimum std for MPPI
            max_std: Maximum std for MPPI
            discount_denom: Denominator for discount factor calculation
            discount_min: Minimum discount factor
            discount_max: Maximum discount factor
            episodic: Whether environment is episodic
            mpc: Whether to use MPC planning
            compile: Whether to compile with torch.compile
            episode_length: Expected episode length
            seed: Random seed
        """
        super().__init__()

        # Store parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.batch_size = batch_size
        self.device_str = device
        self.device = torch.device(device)
        self.episode_length = episode_length

        # Set seed
        set_seed(seed)

        # Create config dict compatible with TD_MPC2_repo
        # Use default model_size=5 parameters if hidden_dim not provided
        if hidden_dim is None:
            enc_dim = 256
            mlp_dim = 512
            num_enc_layers = 2
        else:
            # Extract dimensions from hidden_dim dict
            enc_dim = (
                hidden_dim.get("encoder", [256, 256, 256])[0]
                if isinstance(hidden_dim.get("encoder"), list)
                else 256
            )
            mlp_dim = (
                hidden_dim.get("encoder", [512, 512, 512])[0]
                if isinstance(hidden_dim.get("encoder"), list)
                else 512
            )
            num_enc_layers = (
                len(hidden_dim.get("encoder", [256, 256]))
                if isinstance(hidden_dim.get("encoder"), list)
                else 2
            )

        # Create OmegaConf config
        cfg_dict = {
            "task": "hockey",  # Dummy task name
            "obs": "state",
            "episodic": episodic,
            "steps": 10_000_000,  # Dummy, not used
            "batch_size": batch_size,
            "reward_coef": reward_coef,
            "value_coef": value_coef,
            "termination_coef": termination_coef,
            "consistency_coef": consistency_coef,
            "rho": rho,
            "lr": lr,
            "enc_lr_scale": enc_lr_scale,
            "grad_clip_norm": grad_clip_norm,
            "tau": tau,
            "discount_denom": discount_denom,
            "discount_min": discount_min,
            "discount_max": discount_max,
            "buffer_size": capacity,
            "exp_name": "default",
            "data_dir": None,
            "mpc": mpc,
            "iterations": num_iterations,
            "num_samples": num_samples,
            "num_elites": num_elites,
            "num_pi_trajs": num_pi_trajs,
            "horizon": horizon,
            "min_std": min_std,
            "max_std": max_std,
            "temperature": temperature,
            "log_std_min": log_std_min,
            "log_std_max": log_std_max,
            "entropy_coef": entropy_coef,
            "num_bins": num_bins,
            "vmin": vmin,
            "vmax": vmax,
            "model_size": None,  # Will use explicit dimensions
            "num_enc_layers": num_enc_layers,
            "enc_dim": enc_dim,
            "num_channels": 32,  # Not used for state obs
            "mlp_dim": mlp_dim,
            "latent_dim": latent_dim,
            "task_dim": 0,  # Single task
            "num_q": num_q,
            "dropout": 0.01,
            "simnorm_dim": 8,
            "compile": compile,
            "save_video": False,
            "save_agent": True,
            "seed": seed,
            "work_dir": None,
            "task_title": "Hockey",
            "multitask": False,
            "tasks": ["hockey"],
            "obs_shape": (obs_dim,),
            "action_dim": action_dim,
            "episode_length": episode_length,
            "obs_shapes": [(obs_dim,)],
            "action_dims": [action_dim],
            "episode_lengths": [episode_length],
            "seed_steps": 0,
            "bin_size": (vmax - vmin) / (num_bins - 1),
        }

        # Convert to OmegaConf and parse
        cfg = OmegaConf.create(cfg_dict)
        self.cfg = parse_cfg(cfg)

        # Initialize TD_MPC2_repo agent
        self.tdmpc2 = TDMPC2Repo(self.cfg)

        # Replace buffer with TDMPC2ReplayBuffer
        self.buffer = TDMPC2ReplayBuffer(
            max_size=capacity,
            horizon=horizon,
            batch_size=batch_size,
            use_torch_tensors=True,
            device=device,
            multitask=False,
        )

        # Track episode start for t0 flag
        self._episode_step = 0
        self._last_t0 = True

    def act(self, state, deterministic=False):
        """
        Returns the action for a given state according to the current policy.

        Args:
            state: Observation array
            deterministic: If True, use mean action (no exploration)

        Returns:
            Action array
        """
        t0 = self._last_t0
        self._last_t0 = False

        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            obs = torch.from_numpy(state).float()
        else:
            obs = torch.as_tensor(state, dtype=torch.float32)

        obs = obs.to(self.device, non_blocking=True)
        action = self.tdmpc2.act(obs, t0=t0, eval_mode=deterministic)

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Ensure it's a 1D array
        if action.ndim > 1:
            action = action.flatten()

        return action

    def act_batch(self, states, deterministic=False, t0s=None):
        """
        Process a batch of states at once (for vectorized environments).

        Args:
            states: Batch of observations (batch_size, obs_dim)
            deterministic: If True, use mean actions
            t0s: Optional array of booleans indicating episode starts

        Returns:
            Batch of actions (batch_size, action_dim)
        """
        if t0s is None:
            t0s = np.zeros(len(states), dtype=bool)

        # Convert to torch tensor if needed
        if isinstance(states, np.ndarray):
            obs = torch.from_numpy(states).float()
        else:
            obs = torch.as_tensor(states, dtype=torch.float32)

        obs = obs.to(self.device, non_blocking=True)
        actions = []

        for i, (obs_i, t0) in enumerate(zip(obs, t0s)):
            action = self.tdmpc2.act(obs_i, t0=bool(t0), eval_mode=deterministic)
            actions.append(action)

        actions = torch.stack(actions)

        # Convert to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return actions

    def evaluate(self, state):
        """
        Returns the value of a given state.

        Args:
            state: Observation array

        Returns:
            Estimated state value
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            obs = torch.from_numpy(state).float()
        else:
            obs = torch.as_tensor(state, dtype=torch.float32)

        obs = obs.to(self.device, non_blocking=True)

        with torch.no_grad():
            z = self.tdmpc2.model.encode(obs.unsqueeze(0))
            action, _ = self.tdmpc2.model.pi(z)
            q_value = self.tdmpc2.model.Q(z, action, return_type="avg")
            value = q_value.item()

        return value

    def train(self, steps=1):
        """
        Performs `steps` gradient steps.

        Args:
            steps: Number of training steps to perform

        Returns:
            Dictionary of training statistics (optional, for logging)
        """
        if self.buffer.size < self.horizon + 1:
            return {}

        all_stats = []

        for _ in range(steps):
            try:
                # Sample from buffer
                obs, action, reward, terminated, task = self.buffer.sample(
                    batch_size=self.batch_size, horizon=self.horizon
                )

                # Convert to torch tensors if needed
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float()
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action).float()
                if isinstance(reward, np.ndarray):
                    reward = torch.from_numpy(reward).float()
                if isinstance(terminated, np.ndarray):
                    terminated = torch.from_numpy(terminated).float()

                # Move to device
                obs = obs.to(self.device, non_blocking=True)
                action = action.to(self.device, non_blocking=True)
                reward = reward.to(self.device, non_blocking=True)
                terminated = terminated.to(self.device, non_blocking=True)

                # TD_MPC2_repo expects (horizon+1, batch_size, obs_dim) format
                # Our buffer returns (batch_size, horizon+1, obs_dim)
                obs = obs.permute(1, 0, 2)  # (horizon+1, batch_size, obs_dim)
                action = action.permute(1, 0, 2)  # (horizon, batch_size, action_dim)
                reward = reward.permute(1, 0, 2)  # (horizon, batch_size, 1)
                terminated = terminated.permute(1, 0, 2)  # (horizon, batch_size, 1)

                # Update agent
                stats = self.tdmpc2._update(obs, action, reward, terminated, task=None)

                if stats is not None:
                    # Convert stats to dict if needed
                    if hasattr(stats, "items"):
                        all_stats.append(dict(stats))
                    else:
                        all_stats.append({})

            except RuntimeError as e:
                # Buffer might not have enough data
                if "no episode has length" in str(e):
                    break
                raise

        # Aggregate stats
        if all_stats:
            aggregated = {}
            for key in all_stats[0].keys():
                values = [s.get(key) for s in all_stats if key in s]
                if values:
                    if isinstance(values[0], torch.Tensor):
                        aggregated[key] = torch.stack(values).mean().item()
                    else:
                        aggregated[key] = np.mean(values)
            return aggregated

        return {}

    def save(self, filepath):
        """
        Saves the agent's model to the specified filepath.

        Args:
            filepath: Path to save checkpoint
        """
        self.tdmpc2.save(filepath)

    def load(self, filepath):
        """
        Loads the agent's model from the specified filepath.

        Args:
            filepath: Path to load checkpoint from
        """
        self.tdmpc2.load(filepath)

    def on_episode_start(self, episode):
        """
        Hook that is called at the start of each episode.
        """
        self._episode_step = 0
        self._last_t0 = True

    def on_episode_end(self, episode):
        """
        Hook that is called at the end of each episode.
        """
        self._episode_step = 0
        self._last_t0 = True

    def log_architecture(self):
        """
        Logs the network architecture for the agent.
        """
        return str(self.tdmpc2.model)
