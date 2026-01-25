"""
Wrapper for TD_MPC2_repo's TDMPC2 to integrate with the common Agent interface.

Performance vs rl_hockey TD_MPC2 implementation:
- act_batch loops over envs and calls tdmpc2.act() once per env (no batching).
  The repo uses a single _prev_mean; batching would require per-env state.
- Your impl batches encode and uses per-env planners, so it scales better with
  num_envs > 1.
- Repo compiles _plan and _update with torch.compile(reduce-overhead). If
  "skipping cudagraphs due to cpu device" appears, compile adds overhead
  without benefit. Set "compile": false in agent hyperparameters to avoid that.
- Repo uses episodic + termination prediction and more structure; your impl
  compiles only encoder/dynamics/reward and uses a simpler MPPI setup.
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

# Lazy imports - only import TD_MPC2_repo modules when wrapper is actually used
# This prevents import errors when TDMPC2_REPO is not being used
_imported_modules = None


def _import_tdmpc2_modules():
    """Lazy import of TD_MPC2_repo modules. Called only when wrapper is instantiated."""
    global _imported_modules
    if _imported_modules is not None:
        return _imported_modules

    # Check for required dependencies first
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError(
            "omegaconf is required for TDMPC2_REPO but is not installed.\n"
            "Please install it with: pip install omegaconf\n"
            "Or: conda install -c conda-forge omegaconf"
        )

    # Monkey-patch hydra before importing parser to avoid dependency
    try:
        import hydra  # noqa: F401
    except ImportError:
        # Create a minimal hydra mock
        class HydraMock:
            class utils:
                @staticmethod
                def get_original_cwd():
                    import os

                    return os.getcwd()

        import sys as _sys

        _sys.modules["hydra"] = HydraMock()
        _sys.modules["hydra.utils"] = HydraMock.utils

    try:
        import common.math as math_module
        from common.parser import cfg_to_dataclass, parse_cfg
        from common.seed import set_seed
        from tdmpc2 import TDMPC2 as TDMPC2Repo

        # Patch TensorDict to handle version compatibility issues
        try:
            import torch
            from tensordict import TensorDict

            # Removed TensorDict.__init__ patch as it was causing object corruption
            # (missing _tensordict attribute) in newer versions of tensordict.

            # Patch TensorDict.update to handle version compatibility issues
            if hasattr(TensorDict, "update"):
                _original_tensordict_update = TensorDict.update

                def _patched_tensordict_update(self, input_dict_or_td, **kwargs):
                    """Patched update method that handles _batch_size attribute errors."""
                    # If input is a regular dict, handle it directly
                    if isinstance(input_dict_or_td, dict):
                        # Regular dict - merge directly
                        for key, value in input_dict_or_td.items():
                            try:
                                self[key] = value
                            except (AttributeError, RuntimeError, TypeError):
                                # If assignment fails, try to convert to tensor first
                                if isinstance(value, torch.Tensor):
                                    try:
                                        self.set(key, value)
                                    except (AttributeError, RuntimeError):
                                        # Skip if all methods fail
                                        pass
                        return self

                    # For TensorDict or other types, try original method first
                    try:
                        return _original_tensordict_update(
                            self, input_dict_or_td, **kwargs
                        )
                    except AttributeError as e:
                        error_str = str(e)
                        if (
                            "_batch_size" in error_str
                            or "_device" in error_str
                            or ("batch_size" in error_str and "TensorDict" in error_str)
                        ):
                            # Convert TensorDict to dict and merge manually
                            update_dict = {}
                            try:
                                if isinstance(input_dict_or_td, dict):
                                    update_dict = input_dict_or_td
                                elif hasattr(input_dict_or_td, "to_dict"):
                                    update_dict = input_dict_or_td.to_dict()
                                else:
                                    # Try to convert to dict safely
                                    update_dict = dict(input_dict_or_td)
                            except Exception:
                                # Last resort: try to iterate keys if possible
                                try:
                                    if hasattr(input_dict_or_td, "keys"):
                                        update_dict = {
                                            k: input_dict_or_td[k]
                                            for k in input_dict_or_td.keys()
                                        }
                                except Exception:
                                    pass

                            # Merge into self using direct assignment
                            for key, value in update_dict.items():
                                try:
                                    self[key] = value
                                except (AttributeError, RuntimeError, TypeError):
                                    # If direct assignment fails, try alternative approach
                                    if isinstance(value, torch.Tensor):
                                        try:
                                            self.set(key, value)
                                        except (AttributeError, RuntimeError):
                                            # Skip this key if all methods fail
                                            pass
                            return self
                        raise

                # Apply patch
                TensorDict.update = _patched_tensordict_update

            # Patch termination_statistics to return a regular dict instead of TensorDict
            # This prevents TensorDict compatibility issues when updating info
            _original_termination_statistics = math_module.termination_statistics

            def _patched_termination_statistics(pred, target, eps=1e-9):
                """Patched to return regular dict instead of TensorDict."""
                # Compute statistics directly without creating TensorDict
                pred = pred.squeeze(-1)
                target = target.squeeze(-1)
                rate = target.sum() / len(target)
                tp = ((pred > 0.5) & (target == 1)).sum()
                fn = ((pred <= 0.5) & (target == 1)).sum()
                fp = ((pred > 0.5) & (target == 0)).sum()
                eps_val = 1e-9
                recall = tp / (tp + fn + eps_val)
                precision = tp / (tp + fp + eps_val)
                f1 = 2 * (precision * recall) / (precision + recall + eps_val)
                # Return as regular dict - the update method will handle it
                return {
                    "termination_rate": rate,
                    "termination_f1": f1,
                }

            math_module.termination_statistics = _patched_termination_statistics

            # Patch update_pi to return a regular dict instead of TensorDict
            _original_update_pi = TDMPC2Repo.update_pi

            def _patched_update_pi(self, zs, task):
                """Patched to return regular dict instead of TensorDict."""
                # Call original method
                result = _original_update_pi(self, zs, task)
                # Convert TensorDict to regular dict
                pi_dict = {}
                try:
                    if hasattr(result, "to_dict"):
                        pi_dict = result.to_dict()
                    elif isinstance(result, dict):
                        pi_dict = result
                    else:
                        pi_dict = dict(result)
                except Exception:
                    # Fallback
                    try:
                        if hasattr(result, "keys"):
                            pi_dict = {k: result[k] for k in result.keys()}
                    except Exception:
                        pass
                return pi_dict if pi_dict else result

            TDMPC2Repo.update_pi = _patched_update_pi

        except (ImportError, AttributeError, TypeError) as patch_err:
            # If patching fails, continue anyway - the error will be caught in train()
            import warnings

            warnings.warn(
                f"Failed to patch TensorDict for compatibility: {patch_err}. "
                f"Some features may not work correctly."
            )

    except ImportError as e:
        # If imports fail, provide helpful error message
        raise ImportError(
            f"Failed to import TD_MPC2_repo modules.\n"
            f"Error: {e}\n\n"
            f"Required dependencies for TDMPC2_REPO:\n"
            f"  - omegaconf (for config management)\n"
            f"  - tensordict (for tensor operations)\n"
            f"  - torch (PyTorch)\n\n"
            f"Install missing dependencies with:\n"
            f"  pip install omegaconf tensordict\n"
            f"Or:\n"
            f"  conda install -c conda-forge omegaconf tensordict\n\n"
            f"Also ensure the TD_MPC2_repo/tdmpc2 directory exists and contains all required modules."
        )

    _imported_modules = {
        "OmegaConf": OmegaConf,
        "cfg_to_dataclass": cfg_to_dataclass,
        "parse_cfg": parse_cfg,
        "set_seed": set_seed,
        "TDMPC2Repo": TDMPC2Repo,
    }
    return _imported_modules


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
        win_reward_bonus: float = 10.0,
        win_reward_discount: float = 0.92,
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
            win_reward_bonus: Bonus reward to add to each step in a winning episode.
                Applied with discount factor backwards through the episode. Set to 0.0 to disable.
            win_reward_discount: Discount factor for applying win reward bonus
                backwards through the episode (1.0 = no discount, 0.99 = standard).
        """
        super().__init__()

        # Lazy import TD_MPC2_repo modules (only when wrapper is actually used)
        modules = _import_tdmpc2_modules()
        OmegaConf = modules["OmegaConf"]
        parse_cfg = modules["parse_cfg"]
        set_seed = modules["set_seed"]
        TDMPC2Repo = modules["TDMPC2Repo"]

        # Normalize device: "cuda" -> "cuda:0" so wrapper and inner repo agree
        if isinstance(device, str) and device == "cuda" and torch.cuda.is_available():
            device = "cuda:0"
        self.device_str = device
        self.device = torch.device(device)

        # Store parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.batch_size = batch_size
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
            "obs_shape": {"state": (obs_dim,)},  # Must be a dict, not a tuple
            "action_dim": action_dim,
            "episode_length": episode_length,
            "obs_shapes": [(obs_dim,)],
            "action_dims": [action_dim],
            "episode_lengths": [episode_length],
            "seed_steps": 0,
            "bin_size": (vmax - vmin) / (num_bins - 1),
            "device": self.device_str,
        }

        # Convert to OmegaConf and parse
        cfg = OmegaConf.create(cfg_dict)
        self.cfg = parse_cfg(cfg)

        # Initialize TD_MPC2_repo agent (uses cfg.device, no longer hardcoded cuda:0)
        self.tdmpc2 = TDMPC2Repo(self.cfg)

        # Log device so we can verify GPU is used when expected
        print(f"[TDMPC2RepoWrapper] device={self.device} (model and buffer use this)")

        # Store reference to modules for later use
        self._modules = modules

        # Buffer storage device: default to CPU for larger capacity and negligible transfer overhead
        buffer_device = config.get("buffer_device", "cpu")
        
        # Replace buffer with TDMPC2ReplayBuffer (use same normalized device as model)
        self.buffer = TDMPC2ReplayBuffer(
            max_size=capacity,
            horizon=horizon,
            batch_size=batch_size,
            use_torch_tensors=True,
            device=self.device_str,  # Where sampled batches go (for training)
            buffer_device=buffer_device,  # Where episode data is stored
            pin_memory=(buffer_device == "cpu"),  # Pin memory for faster CPU->GPU transfer
            multitask=False,
            win_reward_bonus=win_reward_bonus,
            win_reward_discount=win_reward_discount,
        )

        # Track episode start for t0 flag
        self._episode_step = 0
        self._last_t0 = True

    def act(self, state, deterministic=False, t0=None):
        """
        Returns the action for a given state according to the current policy.

        Args:
            state: Observation array
            deterministic: If True, use mean action (no exploration)
            t0: Optional bool indicating if this is the first step of an episode.
                If None, uses internal tracking.

        Returns:
            Action array
        """
        if t0 is None:
            t0 = self._last_t0

        # After using t0, next step won't be t0 (unless explicitly set)
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

        Fast path for batch size 1 (num_envs=1): single encode+plan, no loop.
        For batch size > 1, loops over envs (repo uses single _prev_mean).

        Args:
            states: Batch of observations (batch_size, obs_dim)
            deterministic: If True, use mean actions
            t0s: Optional array of booleans indicating episode starts

        Returns:
            Batch of actions (batch_size, action_dim)
        """
        n = len(states) if hasattr(states, "__len__") else states.shape[0]
        if t0s is None:
            t0s = np.zeros(n, dtype=bool)

        if isinstance(states, np.ndarray):
            obs = torch.from_numpy(states).float()
        else:
            obs = torch.as_tensor(states, dtype=torch.float32)
        obs = obs.to(self.device, non_blocking=True)

        if n == 1:
            action = self.tdmpc2.act(
                obs[0], t0=bool(t0s[0]), eval_mode=deterministic
            )
            out = action.cpu().numpy()
            return out[None, :] if out.ndim == 1 else out

        actions = []
        for obs_i, t0 in zip(obs, t0s):
            a = self.tdmpc2.act(obs_i, t0=bool(t0), eval_mode=deterministic)
            actions.append(a)
        actions = torch.stack(actions)
        return actions.cpu().numpy()

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
            # Buffer doesn't have enough data yet - return empty dict
            # This is normal during warmup period
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
                try:
                    stats = self.tdmpc2._update(
                        obs, action, reward, terminated, task=None
                    )
                except (AttributeError, RuntimeError, TypeError) as e:
                    # Handle TensorDict version compatibility issues
                    error_str = str(e)
                    if (
                        any(
                            attr in error_str
                            for attr in [
                                "_batch_size",
                                "_device",
                                "batch_size",
                                "device",
                            ]
                        )
                        and "TensorDict" in error_str
                    ):
                        # TensorDict version compatibility issue
                        import warnings

                        warnings.warn(
                            f"TensorDict compatibility issue encountered: {e}. "
                            f"Skipping this training step. The patches should have "
                            f"prevented this - check tensordict version compatibility."
                        )
                        # Skip this training step
                        continue
                    else:
                        # Re-raise if it's not a TensorDict compatibility issue
                        raise

                if stats is not None:
                    # Convert TensorDict to regular dict and extract scalar values
                    if hasattr(stats, "items"):
                        stats_dict = {}
                        for key, value in stats.items():
                            # Convert tensor to Python scalar
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    stats_dict[key] = value.item()
                                else:
                                    # If it's a multi-element tensor, take mean
                                    stats_dict[key] = value.mean().item()
                            else:
                                stats_dict[key] = value
                        all_stats.append(stats_dict)
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
