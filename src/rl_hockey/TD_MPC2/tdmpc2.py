# TD-MPC2: A model-based reinforcement learning approach to hockey player tracking

import copy
import logging

import torch
import torch.nn.functional as F

# Enable TF32 for better performance on Ampere+ GPUs (RTX 30xx, A100, etc.)
# This speeds up float32 matrix multiplications with minimal accuracy impact
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from rl_hockey.common.agent import Agent
from rl_hockey.TD_MPC2.model_dynamics_simple import DynamicsSimple
from rl_hockey.TD_MPC2.model_encoder import Encoder
from rl_hockey.TD_MPC2.model_policy import Policy
from rl_hockey.TD_MPC2.model_q_ensemble import QEnsemble
from rl_hockey.TD_MPC2.model_reward import Reward
from rl_hockey.TD_MPC2.mppi_planner_simple import MPPIPlannerSimplePaper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


# https://arxiv.org/pdf/2310.16828
# here all the models come together for the workflow
class TDMPC2(Agent):
    def __init__(
        self,
        obs_dim=18,
        action_dim=8,
        latent_dim=512,
        hidden_dim=[256, 256, 256],
        num_q=5,
        simnorm_temperature=1.0,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        gamma=0.99,
        lambda_coef=0.95,
        policy_alpha=1.0,
        policy_beta=1.0,
        horizon=5,
        num_samples=512,
        num_iterations=6,
        capacity=1000000,
        temperature=0.5,
        batch_size=256,
        device="cuda",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_q = num_q
        self.simnorm_temperature = simnorm_temperature
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lr = lr
        self.gamma = gamma
        self.lambda_coef = lambda_coef
        self.policy_alpha = policy_alpha
        self.policy_beta = policy_beta
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.capacity = capacity
        self.device = torch.device(device)

        # Config dict for compatibility with training system
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
            "policy_alpha": policy_alpha,
            "policy_beta": policy_beta,
        }

        # Initialize networks
        self.encoder = Encoder(
            obs_dim,
            latent_dim,
            hidden_dim,
            simnorm_temperature=simnorm_temperature,
        ).to(self.device)
        self.dynamics = DynamicsSimple(
            latent_dim,
            action_dim,
            hidden_dim,
            simnorm_temperature=simnorm_temperature,
        ).to(self.device)
        self.reward = Reward(latent_dim, action_dim, hidden_dim).to(self.device)
        self.q_ensemble = QEnsemble(num_q, latent_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.target_q_ensemble = copy.deepcopy(self.q_ensemble)
        self.policy = Policy(
            latent_dim,
            action_dim,
            hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward.parameters()),
            lr=self.lr,
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q_ensemble.parameters()),
            lr=self.lr,
        )
        self.pi_optimizer = torch.optim.Adam(list(self.policy.parameters()), lr=self.lr)

        # Initialize planner
        self.planner = MPPIPlannerSimplePaper(
            self.dynamics,
            self.reward,
            self.target_q_ensemble,
            self.policy,
            horizon,
            num_samples,
            num_iterations,
            temperature,
            self.gamma,
        )

        # Cache parameter lists for gradient clipping (avoids recreation each step)
        self._model_params = (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward.parameters())
        )

        # Replace buffer with GPU-stored buffer for zero-copy sampling (eliminates CPU->GPU transfer)
        from rl_hockey.common.buffer import ReplayBuffer

        # Store buffer directly on GPU - much faster sampling with no transfer overhead
        self.buffer = ReplayBuffer(
            max_size=capacity, use_torch_tensors=True, device=self.device
        )

        # Pre-allocate tensors on GPU for batch processing (eliminates per-step allocation)
        # These tensors are reused each training step for faster data transfer
        batch_size = self.config.get("batch_size", 256)
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

        # Compile models for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, "compile"):
                self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
                self.dynamics = torch.compile(self.dynamics, mode="reduce-overhead")
                self.reward = torch.compile(self.reward, mode="reduce-overhead")
                self.q_ensemble = torch.compile(self.q_ensemble, mode="reduce-overhead")
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
        except Exception:
            # Fallback if torch.compile not available or fails
            pass
        # Ensure planner buffers (e.g., gamma_powers) are on the correct device
        self.planner.to(self.device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """
        Select action using MPC planning.

        Args:
            obs: (obs_dim,) observation
            deterministic: if True, return mean action

        Returns:
            action: (action_dim,) planned action
        """
        obs = torch.FloatTensor(obs).to(self.device)

        z = self.encoder(obs.unsqueeze(0)).squeeze(0)

        action = self.planner.plan(z, return_mean=deterministic)

        return action.cpu().numpy()

    @torch.no_grad()
    def act_batch(self, obs_batch, deterministic=False):
        """
        Select actions for batch of observations (for vectorized environments).

        Args:
            obs_batch: (batch_size, obs_dim) observations
            deterministic: if True, return mean actions

        Returns:
            actions: (batch_size, action_dim) planned actions
        """
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)

        z_batch = self.encoder(obs_batch)

        # Plan actions for each state in batch
        actions = []
        for z in z_batch:
            action = self.planner.plan(z, return_mean=deterministic)
            actions.append(action)

        return torch.stack(actions).cpu().numpy()

    def evaluate(self, obs):
        """
        Evaluate state value using Q-ensemble.

        Args:
            obs: (obs_dim,) observation

        Returns:
            value: scalar state value
        """
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            z = self.encoder(obs.unsqueeze(0))

            # Use policy to get action
            action = self.policy.mean_action(z)

            # Get minimum Q-value from ensemble
            q_value = self.q_ensemble.min(z, action)

            return q_value.item()

    def train(self, steps=1, enable_anomaly_detection=False):
        """
        Train all networks for given number of steps.

        Args:
            steps: number of gradient steps
            enable_anomaly_detection: if True, enable PyTorch anomaly detection for debugging

        Returns:
            dict: losses for logging
        """
        # Enable anomaly detection for better error tracking (helps identify inplace operations)
        if enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.warning(
                "Anomaly detection enabled - this will slow down training significantly"
            )

        all_losses = {
            "dynamics_loss": [],
            "reward_loss": [],
            "value_loss": [],
            "q_loss": [],
            "policy_loss": [],
            "total_loss": [],
            "loss": [],  # Alias for train_run.py compatibility
        }

        try:
            for _ in range(steps):
                used_sequences = False
                batch_size = self.config.get("batch_size", 256)

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

                    batch_size, horizon_plus_one, _ = obs_seq.shape
                    horizon = horizon_plus_one - 1

                    obs_flat = obs_seq.reshape(batch_size * (horizon + 1), -1)
                    z_flat = self.encoder(obs_flat)
                    z_seq = z_flat.reshape(batch_size, horizon + 1, self.latent_dim)

                    # Clone to avoid view issues
                    z_pred = z_seq[:, 0].clone()

                    q_params = list(self.q_ensemble.parameters())
                    q_requires = [p.requires_grad for p in q_params]
                    for p in q_params:
                        p.requires_grad = False

                    dynamics_loss = 0.0
                    reward_loss = 0.0
                    value_loss = 0.0

                    for t in range(horizon):
                        a_t = actions_seq[:, t]
                        r_t = rewards_seq[:, t]
                        d_t = dones_seq[:, t]

                        z_next_pred = self.dynamics(z_pred, a_t)
                        z_target = z_seq[:, t + 1].detach()

                        weight = self.lambda_coef**t
                        dynamics_loss = dynamics_loss + weight * F.mse_loss(
                            z_next_pred, z_target
                        )

                        r_pred = self.reward(z_pred, a_t)
                        reward_loss = reward_loss + weight * F.mse_loss(r_pred, r_t)

                        q_preds = self.q_ensemble(z_pred, a_t)
                        with torch.no_grad():
                            next_action, _, _ = self.policy.sample(z_next_pred)
                            target_q = self.target_q_ensemble.min_subsample(
                                z_next_pred, next_action, k=2
                            )
                            q_target = r_t + self.gamma * (1 - d_t) * target_q
                        step_value_loss = 0.0
                        for q_pred in q_preds:
                            step_value_loss = step_value_loss + F.mse_loss(
                                q_pred, q_target
                            )
                        value_loss = value_loss + weight * step_value_loss

                        # Clone to avoid inplace modification issues
                        z_pred = z_next_pred.clone()

                    for p, req in zip(q_params, q_requires):
                        p.requires_grad = req

                    used_sequences = True

                if not used_sequences:
                    # Sample from buffer
                    obs, actions, rewards, next_obs, dones = self.buffer.sample(
                        batch_size
                    )

                    # Optimized: buffer is stored directly on GPU - zero-copy sampling!
                    # Samples are already on GPU, just use them directly (no transfer needed)
                    if (
                        isinstance(obs, torch.Tensor)
                        and obs.device.type == self.device.type
                        and (
                            self.device.index is None
                            or obs.device.index == self.device.index
                        )
                    ):
                        # Buffer is on GPU - use samples directly, just ensure shapes are correct
                        if rewards.dim() == 1:
                            rewards = rewards.unsqueeze(-1)
                        if dones.dim() == 1:
                            dones = dones.unsqueeze(-1)
                        # Samples are already on GPU, ready to use - no transfer overhead!
                        # No need for pre-allocated buffers since samples are already in the right place
                    else:
                        # Fallback: buffer on CPU - transfer to GPU using pre-allocated buffers
                        if isinstance(obs, torch.Tensor):
                            # CPU tensors (pinned memory) - transfer to GPU buffers
                            obs = obs.to(self.device, non_blocking=True)
                            actions = actions.to(self.device, non_blocking=True)
                            rewards = rewards.view(-1, 1).to(
                                self.device, non_blocking=True
                            )
                            next_obs = next_obs.to(self.device, non_blocking=True)
                            dones = dones.view(-1, 1).to(self.device, non_blocking=True)
                        else:
                            # Numpy arrays - convert and transfer
                            obs = (
                                torch.from_numpy(obs)
                                .float()
                                .to(self.device, non_blocking=True)
                            )
                            actions = (
                                torch.from_numpy(actions)
                                .float()
                                .to(self.device, non_blocking=True)
                            )
                            rewards = (
                                torch.from_numpy(rewards)
                                .float()
                                .reshape(-1, 1)
                                .to(self.device, non_blocking=True)
                            )
                            next_obs = (
                                torch.from_numpy(next_obs)
                                .float()
                                .to(self.device, non_blocking=True)
                            )
                            dones = (
                                torch.from_numpy(dones)
                                .float()
                                .reshape(-1, 1)
                                .to(self.device, non_blocking=True)
                            )

                    # Encode observations
                    z = self.encoder(obs)
                    with torch.no_grad():
                        z_next_enc = self.encoder(next_obs)

                    q_params = list(self.q_ensemble.parameters())
                    q_requires = [p.requires_grad for p in q_params]
                    for p in q_params:
                        p.requires_grad = False

                    # 1. Dynamics loss
                    z_next_pred = self.dynamics(z, actions)
                    dynamics_loss = F.mse_loss(z_next_pred, z_next_enc.detach())

                    # 2. Reward loss
                    r_pred = self.reward(z, actions)
                    reward_loss = F.mse_loss(r_pred, rewards)

                    # 3. Value loss (TD target)
                    q_preds = self.q_ensemble(z, actions)
                    with torch.no_grad():
                        next_actions, _, _ = self.policy.sample(z_next_pred)
                        target_q = self.target_q_ensemble.min_subsample(
                            z_next_pred, next_actions, k=2
                        )
                        q_target = rewards + self.gamma * (1 - dones) * target_q
                    value_loss = 0.0
                    for q_pred in q_preds:
                        value_loss = value_loss + F.mse_loss(q_pred, q_target)

                    for p, req in zip(q_params, q_requires):
                        p.requires_grad = req

                if used_sequences:
                    z_q = z_seq[:, 0].detach().clone()
                    q_loss = 0.0
                    for t in range(horizon):
                        a_t = actions_seq[:, t]
                        r_t = rewards_seq[:, t]
                        d_t = dones_seq[:, t]
                        with torch.no_grad():
                            z_next_pred = self.dynamics(z_q, a_t)
                            next_actions, _, _ = self.policy.sample(z_next_pred)
                            target_q = self.target_q_ensemble.min_subsample(
                                z_next_pred, next_actions, k=2
                            )
                            q_target = r_t + self.gamma * (1 - d_t) * target_q
                        q_preds = self.q_ensemble(z_q, a_t)
                        step_q_loss = 0.0
                        for q_pred in q_preds:
                            step_q_loss = step_q_loss + F.mse_loss(q_pred, q_target)
                        q_loss = q_loss + (self.lambda_coef**t) * step_q_loss
                        # Clone to avoid inplace modification issues
                        z_q = z_next_pred.detach().clone()
                else:
                    with torch.no_grad():
                        z_next_pred = self.dynamics(z.detach(), actions)
                        next_actions, _, _ = self.policy.sample(z_next_pred)
                        target_q = self.target_q_ensemble.min_subsample(
                            z_next_pred, next_actions, k=2
                        )
                        q_target = rewards + self.gamma * (1 - dones) * target_q
                    q_preds = self.q_ensemble(z.detach(), actions)
                    q_loss = 0.0
                    for q_pred in q_preds:
                        q_loss = q_loss + F.mse_loss(q_pred, q_target)

                if used_sequences:
                    # Clone immediately to avoid view issues from indexing
                    z_pi = z_seq[:, 0].detach().clone()
                else:
                    z_pi = z.detach().clone()

                policy_loss = 0.0
                q_params = list(self.q_ensemble.parameters())
                dyn_params = list(self.dynamics.parameters())
                q_requires = [p.requires_grad for p in q_params]
                dyn_requires = [p.requires_grad for p in dyn_params]
                for p in q_params:
                    p.requires_grad = False
                for p in dyn_params:
                    p.requires_grad = False

                # Ensure z_roll requires gradients for policy loss computation
                # (gradients flow through policy, not dynamics)
                # CRITICAL: Use torch.tensor or clone with requires_grad, NOT .requires_grad_()
                # which modifies inplace and breaks the computation graph
                z_roll = z_pi.clone()
                z_roll.requires_grad = True

                # Store step losses in a list to avoid keeping references to intermediate tensors
                # across loop iterations (prevents inplace modification errors)
                step_losses = []

                for t in range(self.horizon):
                    action, log_prob, _ = self.policy.sample(z_roll)
                    q_val = self.q_ensemble.min(z_roll, action)
                    entropy = -log_prob
                    # Compute step_loss for this iteration
                    step_loss = -(
                        self.policy_alpha * q_val - self.policy_beta * entropy
                    ).mean()
                    # Store weighted step loss (don't accumulate yet to avoid keeping references)
                    step_losses.append((self.lambda_coef**t) * step_loss)

                    # CRITICAL: Detach z_roll and action BEFORE computing next z_roll
                    # This breaks the computation graph and prevents inplace modification errors
                    z_roll_detached = z_roll.detach()
                    action_detached = action.detach()

                    # Compute next z_roll with detached inputs (dynamics has requires_grad=False anyway)
                    with torch.no_grad():
                        z_roll_next = self.dynamics(z_roll_detached, action_detached)

                    # Clone and enable requires_grad WITHOUT inplace operation
                    z_roll = z_roll_next.clone()
                    z_roll.requires_grad = True

                # Sum all step losses after the loop to avoid keeping references during iterations
                policy_loss = sum(step_losses)

                for p, req in zip(q_params, q_requires):
                    p.requires_grad = req
                for p, req in zip(dyn_params, dyn_requires):
                    p.requires_grad = req

                # Optimize Model (Encoder, Dynamics, Reward)
                self.optimizer.zero_grad()
                (dynamics_loss + reward_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self._model_params, max_norm=10.0)
                self.optimizer.step()

                # Optimize Q-network separately
                self.q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.q_ensemble.parameters(), max_norm=10.0
                )
                self.q_optimizer.step()

                # Optimize Policy
                self.pi_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
                self.pi_optimizer.step()

                # Compute total loss for logging only (after all backward passes)
                total_loss = (
                    dynamics_loss.detach()
                    + reward_loss.detach()
                    + value_loss.detach()
                    + q_loss.detach()
                    + policy_loss.detach()
                )

                # Update target network (soft update) - only every step (tau is small)
                self._update_target_network(tau=0.01)

                # Store losses - defer .item() calls to reduce CPU-GPU synchronization overhead
                # Only convert to Python values at the end if we have multiple steps
                if steps == 1:
                    # Single step: convert immediately (required for compatibility)
                    all_losses["dynamics_loss"].append(dynamics_loss.item())
                    all_losses["reward_loss"].append(reward_loss.item())
                    all_losses["value_loss"].append(value_loss.item())
                    all_losses["q_loss"].append(q_loss.item())
                    all_losses["policy_loss"].append(policy_loss.item())
                    all_losses["total_loss"].append(total_loss.item())
                    all_losses["loss"].append(total_loss.item())
                else:
                    # Multiple steps: store tensors and convert in batch at the end
                    all_losses["dynamics_loss"].append(dynamics_loss)
                    all_losses["reward_loss"].append(reward_loss)
                    all_losses["value_loss"].append(value_loss)
                    all_losses["q_loss"].append(q_loss)
                    all_losses["policy_loss"].append(policy_loss)
                    all_losses["total_loss"].append(total_loss)
                    all_losses["loss"].append(total_loss)

            # If we stored tensors, convert them to Python values now (batch operation)
            if steps > 1 and len(all_losses["dynamics_loss"]) > 0:
                if isinstance(all_losses["dynamics_loss"][0], torch.Tensor):
                    # Batch convert all tensors to Python values (more efficient than per-step)
                    all_losses["dynamics_loss"] = [
                        loss.item() for loss in all_losses["dynamics_loss"]
                    ]
                    all_losses["reward_loss"] = [
                        loss.item() for loss in all_losses["reward_loss"]
                    ]
                    all_losses["value_loss"] = [
                        loss.item() for loss in all_losses["value_loss"]
                    ]
                    all_losses["q_loss"] = [
                        loss.item() for loss in all_losses["q_loss"]
                    ]
                    all_losses["policy_loss"] = [
                        loss.item() for loss in all_losses["policy_loss"]
                    ]
                    all_losses["total_loss"] = [
                        loss.item() for loss in all_losses["total_loss"]
                    ]
                    all_losses["loss"] = [loss.item() for loss in all_losses["loss"]]

        finally:
            # Always disable anomaly detection after training (even if error occurred)
            if enable_anomaly_detection:
                torch.autograd.set_detect_anomaly(False)

        return all_losses

    def _update_target_network(self, tau=0.01):
        """Soft update of target Q-network using vectorized operations"""
        # Optimized: Cache parameter lists to avoid repeated attribute access
        # In-place operations are already used for efficiency
        with torch.no_grad():
            source_params = list(self.q_ensemble.parameters())
            target_params = list(self.target_q_ensemble.parameters())
            for param, target_param in zip(source_params, target_params):
                # In-place operation is faster than copy_
                target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)

    def _get_state_dict(self, model):
        """
        Get state dict from model, handling compiled models.
        Returns the original module's state_dict if model is compiled.
        """
        if hasattr(model, "_orig_mod"):
            # Model is compiled, get state_dict from original module
            return model._orig_mod.state_dict()
        else:
            # Model is not compiled, get state_dict directly
            return model.state_dict()

    def save(self, path):
        """Save agent"""
        torch.save(
            {
                "encoder": self._get_state_dict(self.encoder),
                "dynamics": self._get_state_dict(self.dynamics),
                "reward": self._get_state_dict(self.reward),
                "q_ensemble": self._get_state_dict(self.q_ensemble),
                "policy": self._get_state_dict(self.policy),
                "optimizer": self.optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "pi_optimizer": self.pi_optimizer.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "num_q": self.num_q,
            },
            path,
        )

    def _load_state_dict_compatible(self, model, state_dict):
        """
        Load state dict into model, handling both compiled and uncompiled models.

        If model is compiled (has _orig_mod) and checkpoint doesn't have _orig_mod prefix,
        load into the original module. Otherwise, load directly.
        """
        # Check if model is compiled
        is_compiled = hasattr(model, "_orig_mod")

        # Check if state_dict keys have _orig_mod prefix
        has_prefix = any(key.startswith("_orig_mod.") for key in state_dict.keys())

        if is_compiled and not has_prefix:
            # Model is compiled but checkpoint is not - load into original module
            model._orig_mod.load_state_dict(state_dict)
        else:
            # Either model is not compiled, or checkpoint has matching prefix
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                # If loading fails, try loading into original module if available
                if is_compiled and "Missing key(s)" in str(e):
                    logger.warning(
                        f"Failed to load state dict directly, trying original module: {e}"
                    )
                    model._orig_mod.load_state_dict(state_dict)
                else:
                    raise

    def load(self, path):
        """Load agent"""
        # Load on CPU first to avoid GPU memory issues when loading in parallel processes
        checkpoint = torch.load(path, map_location="cpu")

        # Load all models with compatibility handling
        self._load_state_dict_compatible(self.encoder, checkpoint["encoder"])
        self._load_state_dict_compatible(self.dynamics, checkpoint["dynamics"])
        self._load_state_dict_compatible(self.reward, checkpoint["reward"])
        self._load_state_dict_compatible(self.q_ensemble, checkpoint["q_ensemble"])
        self._load_state_dict_compatible(self.policy, checkpoint["policy"])

        # Load optimizer
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "q_optimizer" in checkpoint:
                self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            else:
                logger.warning(
                    "No q_optimizer found in checkpoint, initializing new optimizer"
                )
            self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

        # Update target network
        self._load_state_dict_compatible(
            self.target_q_ensemble, checkpoint["q_ensemble"]
        )
