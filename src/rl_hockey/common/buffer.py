import random

import numpy as np
import torch

from rl_hockey.common.reward_backprop import apply_win_reward_backprop
from rl_hockey.common.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    def __init__(
        self,
        max_size=1_000_000,
        use_torch_tensors=False,
        pin_memory=False,
        device="cpu",
        normalize_obs = False
    ):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum size of the buffer
            use_torch_tensors: If True, use torch tensors instead of numpy arrays (faster for GPU transfers)
            pin_memory: If True, pin memory for faster CPU->GPU transfers (only if use_torch_tensors=True and device="cpu")
            device: Device to store buffer on ("cpu" or "cuda"). If "cuda", buffer is stored directly on GPU for zero-copy sampling.
        """
        self.max_size = max_size
        self.current_idx = 0
        self.size = 0
        self.use_torch_tensors = use_torch_tensors or (device != "cpu")
        self.device = (
            device
            if isinstance(device, str)
            else (device.type if hasattr(device, "type") else "cpu")
        )
        # Only use pinned memory if on CPU (pinned memory doesn't apply to GPU tensors)
        self.pin_memory = (
            pin_memory
            and self.device == "cpu"
            and self.use_torch_tensors
            and torch.cuda.is_available()
        )

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = None
        self.normalize_obs = normalize_obs

        self.obs_mean = None
        self.obs_M2 = None

    def normalize(self, state: np.ndarray):
        normalized_state = (state - self.obs_mean) / (np.sqrt(self.obs_M2 / self.size) + 1e-8)
        return normalized_state


    def update_obs_stats(self, state: np.ndarray):
        delta = state - self.obs_mean
        self.obs_mean += delta / self.size
        delta2 = state - self.obs_mean
        self.obs_M2 += delta * delta2


    def store(self, transition):
        state, action, reward, next_state, done = transition

        if self.size == 0:
            if self.use_torch_tensors:
                # Use torch tensors - store directly on GPU if device="cuda" (zero-copy sampling)
                # Or on CPU with pinned memory for faster transfers
                device_kwargs = (
                    {"device": self.device}
                    if self.device != "cpu"
                    else {"pin_memory": self.pin_memory}
                )
                self.state = torch.empty(
                    (self.max_size, len(state)), dtype=torch.float32, **device_kwargs
                )
                self.action = torch.empty(
                    (self.max_size, len(action)), dtype=torch.float32, **device_kwargs
                )
                self.next_state = torch.empty(
                    (self.max_size, len(state)), dtype=torch.float32, **device_kwargs
                )
                self.reward = torch.empty(
                    (self.max_size, 1), dtype=torch.float32, **device_kwargs
                )
                self.done = torch.empty(
                    (self.max_size, 1), dtype=torch.float32, **device_kwargs
                )
            else:
                # Original numpy arrays (for backward compatibility)
                self.state = np.empty((self.max_size, len(state)), dtype=np.float32)
                self.action = np.empty((self.max_size, len(action)), dtype=np.float32)
                self.next_state = np.empty(
                    (self.max_size, len(state)), dtype=np.float32
                )
                self.reward = np.empty((self.max_size, 1), dtype=np.float32)
                self.done = np.empty((self.max_size, 1), dtype=np.float32)

            self.obs_mean = np.zeros_like(state, dtype=np.float32)
            self.obs_M2 = np.zeros_like(state, dtype=np.float32)

        # Update observation statistics
        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.update_obs_stats(state)
        self.update_obs_stats(next_state)

        # Store data (convert to tensor if using torch tensors)
        if self.use_torch_tensors:
            # Convert to tensor and move to buffer device if needed
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            action_tensor = torch.as_tensor(action, dtype=torch.float32)
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
            # Handle reward/done - can be scalar or array, ensure shape (1,)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
            if reward_tensor.dim() == 0:
                reward_tensor = reward_tensor.unsqueeze(0)
            reward_tensor = reward_tensor.reshape(1)
            done_tensor = torch.as_tensor(done, dtype=torch.float32)
            if done_tensor.dim() == 0:
                done_tensor = done_tensor.unsqueeze(0)
            done_tensor = done_tensor.reshape(1)

            # Move to buffer device if buffer is on GPU
            if self.device != "cpu":
                state_tensor = state_tensor.to(self.device, non_blocking=True)
                action_tensor = action_tensor.to(self.device, non_blocking=True)
                next_state_tensor = next_state_tensor.to(self.device, non_blocking=True)
                reward_tensor = reward_tensor.to(self.device, non_blocking=True)
                done_tensor = done_tensor.to(self.device, non_blocking=True)

            self.state[self.current_idx] = state_tensor
            self.action[self.current_idx] = action_tensor
            self.next_state[self.current_idx] = next_state_tensor
            self.reward[self.current_idx] = reward_tensor
            self.done[self.current_idx] = done_tensor
        else:
            self.state[self.current_idx] = state
            self.action[self.current_idx] = action
            self.next_state[self.current_idx] = next_state
            self.reward[self.current_idx] = reward
            self.done[self.current_idx] = done


    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size

        if self.use_torch_tensors:
            # Use torch.randint for faster random sampling
            # If buffer is on GPU, generate indices on same device for efficiency
            if self.device != "cpu":
                idx = torch.randint(0, self.size, (batch_size,), device=self.device)
            else:
                idx = torch.randint(0, self.size, (batch_size,))
            
            state = self.state[idx]
            action = self.action[idx]
            reward = self.reward[idx]
            next_state = self.next_state[idx]
            done = self.done[idx]
            

        else:
            idx = np.random.randint(0, self.size, size=batch_size)
            state = self.state[idx]
            action = self.action[idx]
            reward = self.reward[idx]
            next_state = self.next_state[idx]
            done = self.done[idx]

        if self.normalize_obs:
            state = self.normalize(state)
            next_state = self.normalize(next_state)

        return state, action, reward, next_state, done

    def _is_valid_sequence_start(self, start_idx, horizon):
        """Check if a sequence starting at start_idx is valid (no terminals, no wrap)."""
        if self.size < horizon + 1:
            return False

        # If buffer not full, ensure we don't exceed current size
        if self.size < self.max_size:
            if start_idx + horizon >= self.size:
                return False
            idx_range = range(start_idx, start_idx + horizon)
        else:
            # Avoid sequences that cross the circular buffer boundary at current_idx
            if start_idx < self.current_idx:
                if start_idx + horizon >= self.current_idx:
                    return False
            else:
                if start_idx + horizon >= self.max_size:
                    return False
            idx_range = range(start_idx, start_idx + horizon)

        # Ensure no terminal transitions in the sequence
        if self.use_torch_tensors:
            dones = self.done[list(idx_range)]
            return not torch.any(dones > 0.5).item()
        return not np.any(self.done[list(idx_range)] > 0.5)

    def sample_sequences(self, batch_size, horizon):
        """
        Sample contiguous sequences of transitions.

        Returns
        -------
        tuple or None
            (obs_seq, action_seq, reward_seq, done_seq) with shapes:
            obs_seq: (batch, horizon+1, obs_dim)
            action_seq: (batch, horizon, action_dim)
            reward_seq: (batch, horizon, 1)
            done_seq: (batch, horizon, 1)
            Returns None if not enough data.
        """
        if self.size < horizon + 1:
            return None

        max_batch = self.size - horizon
        batch_size = min(batch_size, max_batch)

        starts = []
        attempts = 0
        max_attempts = batch_size * 20
        max_index = self.size if self.size < self.max_size else self.max_size

        while len(starts) < batch_size and attempts < max_attempts:
            start_idx = random.randrange(0, max_index)
            if self._is_valid_sequence_start(start_idx, horizon):
                starts.append(start_idx)
            attempts += 1

        if len(starts) == 0:
            return None

        # Build index matrix without wrap (sequence validity ensures no wrap)
        idx_matrix = [[s + t for t in range(horizon + 1)] for s in starts]

        if self.use_torch_tensors:
            idx_tensor = torch.as_tensor(
                idx_matrix, dtype=torch.long, device=self.state.device
            )
            obs_seq = self.state[idx_tensor]
            action_seq = self.action[idx_tensor[:, :horizon]]
            reward_seq = self.reward[idx_tensor[:, :horizon]]
            done_seq = self.done[idx_tensor[:, :horizon]]
        else:
            idx_array = np.asarray(idx_matrix, dtype=np.int64)
            obs_seq = self.state[idx_array]
            action_seq = self.action[idx_array[:, :horizon]]
            reward_seq = self.reward[idx_array[:, :horizon]]
            done_seq = self.done[idx_array[:, :horizon]]

        return obs_seq, action_seq, reward_seq, done_seq

    def clear(self):
        """Clear the replay buffer, resetting it to empty state."""
        self.current_idx = 0
        self.size = 0
        # Keep arrays allocated but mark as empty


class TDMPC2ReplayBuffer:
    """Episode-based replay buffer for TD-MPC2.

    Stores full episodes and samples contiguous subsequences of length horizon+1
    for world model training. Functionally similar to the torchrl-based Buffer
    in tdmpc2_repo, but implemented in the same style as ReplayBuffer (no
    torchrl/tensordict, optional numpy/torch, explicit device/pin_memory).

    Supports:
    - add_episode: add one episode (obs, action, reward, terminated, task)
    - add_episodes: add multiple episodes (batch load)
    - store: append transitions; when done=True, the current trajectory is
      closed and added as an episode (for compatibility with transition-based
      env loops)
    - sample: returns (obs, action, reward, terminated, task) with shapes
      (B, H+1, obs_dim), (B, H, action_dim), (B, H, 1), (B, H, 1), (B, task_dim) or None
    - sample_sequences: same as sample but returns (obs, action, reward, terminated)
      for drop-in use with rl_hockey TD-MPC2 train (no task in the 4-tuple)
    """

    def __init__(
        self,
        max_size=1_000_000,
        horizon=5,
        batch_size=256,
        use_torch_tensors=False,
        pin_memory=False,
        device="cpu",
        multitask=False,
        win_reward_bonus=10.0,
        win_reward_discount=0.92,
    ):
        """
        Initialize TD-MPC2 replay buffer.

        Args:
            max_size: Maximum number of transitions across all episodes.
                Oldest episodes are evicted when exceeded.
            horizon: Length of sampled subsequences (number of transitions in
                the chunk; obs will have horizon+1 steps).
            batch_size: Number of subsequences returned per sample.
            use_torch_tensors: If True, use torch tensors; else numpy.
            pin_memory: If True, pin memory for faster CPU to GPU transfer
                (only when use_torch_tensors=True and device="cpu").
            device: Device to place sampled batches on ("cpu" or "cuda").
            multitask: If True, expect and return task in add_episode and sample.
            win_reward_bonus: Bonus reward to add to each step in a winning episode.
                Applied with discount factor backwards through the episode.
            win_reward_discount: Discount factor for applying win reward bonus
                backwards through the episode (1.0 = no discount, 0.99 = standard).
        """
        self.max_size = max_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.use_torch_tensors = use_torch_tensors or (device != "cpu")
        self.device = (
            device
            if isinstance(device, str)
            else (device.type if hasattr(device, "type") else "cpu")
        )
        self.pin_memory = (
            pin_memory
            and self.device == "cpu"
            and self.use_torch_tensors
            and torch.cuda.is_available()
        )
        self.multitask = multitask
        self.win_reward_bonus = win_reward_bonus
        self.win_reward_discount = win_reward_discount

        self._episodes = []
        self._total_transitions = 0

        # For store(transition): accumulate into current episode until done
        self._current_obs = []
        self._current_actions = []
        self._current_rewards = []
        self._current_dones = []
        self._current_winner = None

    @property
    def capacity(self):
        """Maximum number of transitions the buffer can hold."""
        return self.max_size

    @property
    def num_eps(self):
        """Number of episodes in the buffer."""
        return len(self._episodes)

    @property
    def size(self):
        """Current number of transitions in the buffer."""
        return self._total_transitions

    def _evict_if_needed(self, extra):
        """Evict oldest episodes until _total_transitions + extra <= max_size."""
        while self._episodes and self._total_transitions + extra > self.max_size:
            ep = self._episodes.pop(0)
            T = ep["action"].shape[0]
            self._total_transitions -= T

    def _add_episode_internal(
        self,
        obs,
        action,
        reward,
        terminated,
        task=None,
        winner=None,
        reward_original=None,
    ):
        """Append one episode to _episodes and update _total_transitions."""
        T = action.shape[0]
        self._evict_if_needed(T)
        ep = {
            "obs": obs,
            "action": action,
            "reward": reward,  # This is the backprop reward (used for training)
            "terminated": terminated,
            "task": task,
            "winner": winner,
            "reward_original": reward_original
            if reward_original is not None
            else reward,  # Original reward before backprop
        }
        self._episodes.append(ep)
        self._total_transitions += T

    def _to_buffer_dtype(self, x, is_numpy=None):
        """Convert to torch or numpy to match buffer config."""
        if is_numpy is None:
            is_numpy = not self.use_torch_tensors
        if is_numpy:
            if hasattr(x, "cpu") and callable(getattr(x, "cpu", None)):
                return np.asarray(x.detach().cpu().numpy(), dtype=np.float32)
            return np.asarray(x, dtype=np.float32)
        t = torch.as_tensor(x, dtype=torch.float32)
        if self.device != "cpu":
            t = t.to(self.device, non_blocking=True)
        elif self.pin_memory and t.is_cpu and torch.cuda.is_available():
            t = t.pin_memory()
        return t

    def store(self, transition, winner=None):
        """Store a transition. When done is True, the current trajectory is
        closed and added as an episode.

        Args:
            transition: (state, action, reward, next_state, done).
            winner: Optional winner information (1 for agent win, -1 for loss, 0 for draw).
                Should be provided when done=True to enable reward shaping for wins.
        """
        state, action, reward, next_state, done = transition

        if len(self._current_obs) == 0:
            self._current_obs.append(
                self._to_buffer_dtype(state)
                if self.use_torch_tensors
                else np.asarray(state, dtype=np.float32)
            )
        self._current_obs.append(
            self._to_buffer_dtype(next_state)
            if self.use_torch_tensors
            else np.asarray(next_state, dtype=np.float32)
        )
        a = (
            self._to_buffer_dtype(action)
            if self.use_torch_tensors
            else np.asarray(action, dtype=np.float32)
        )
        r = np.float32(reward)
        d = np.float32(done)
        if self.use_torch_tensors:
            r = torch.as_tensor(r, dtype=torch.float32)
            d = torch.as_tensor(d, dtype=torch.float32)
        self._current_actions.append(a)
        self._current_rewards.append(r)
        self._current_dones.append(d)

        if done:
            if winner is not None:
                self._current_winner = winner
            self._flush_episode()

    def _flush_episode(self):
        """Build one episode from _current_* and add it; clear _current_*.
        If the episode was a win and win_reward_bonus > 0, applies discounted
        reward bonus backwards through the episode.
        """
        if len(self._current_actions) == 0:
            self._current_obs = []
            self._current_dones = []
            self._current_winner = None
            return

        if self.use_torch_tensors:
            obs = torch.stack(self._current_obs, dim=0)
            action = torch.stack(self._current_actions, dim=0)
            reward = torch.stack(self._current_rewards, dim=0)
            terminated = torch.stack(self._current_dones, dim=0)
        else:
            obs = np.stack(self._current_obs, axis=0)
            action = np.stack(self._current_actions, axis=0)
            reward = np.stack(self._current_rewards, axis=0)
            terminated = np.stack(self._current_dones, axis=0)

        if reward.ndim == 1:
            reward = reward.reshape(-1, 1)
        if terminated.ndim == 1:
            terminated = terminated.reshape(-1, 1)

        # Save original reward before backpropagation
        if self.use_torch_tensors:
            reward_original = reward.clone()
        else:
            reward_original = reward.copy()

        # Apply reward shaping for winning episodes using the backpropagation function
        if (
            self._current_winner == 1
            and self.win_reward_bonus > 0.0
            and len(reward) > 0
        ):
            # Flatten reward for the function (it expects 1D or will flatten)
            reward_flat = reward.flatten()
            # Apply backpropagation
            reward_flat, _, _ = apply_win_reward_backprop(
                reward_flat,
                winner=self._current_winner,
                win_reward_bonus=self.win_reward_bonus,
                win_reward_discount=self.win_reward_discount,
                use_torch=self.use_torch_tensors,
            )
            # Reshape back to original shape
            reward = reward_flat.reshape(reward.shape)

        self._add_episode_internal(
            obs,
            action,
            reward,
            terminated,
            task=None,
            winner=self._current_winner,
            reward_original=reward_original,
        )
        self._current_obs = []
        self._current_actions = []
        self._current_rewards = []
        self._current_dones = []
        self._current_winner = None

    def add_episode(self, obs, action, reward, terminated, task=None, winner=None):
        """Add one episode to the buffer.

        Args:
            obs: (T+1, obs_dim) observations for steps 0..T.
            action: (T, action_dim) actions for transitions 0..T-1.
            reward: (T,) or (T, 1) rewards. This should be the backprop reward if backprop was applied.
            terminated: (T,) or (T, 1) done flags.
            task: Optional (T, task_dim) or (task_dim,) for multitask; ignored if multitask=False.
            winner: Optional winner information (1 for agent win, -1 for loss, 0 for draw).
        """
        obs = self._to_buffer_dtype(obs)
        action = self._to_buffer_dtype(action)
        reward = self._to_buffer_dtype(reward)
        terminated = self._to_buffer_dtype(terminated)
        if reward.ndim == 1:
            reward = reward.reshape(-1, 1)
        if terminated.ndim == 1:
            terminated = terminated.reshape(-1, 1)
        t = task
        if t is not None and self.multitask:
            t = self._to_buffer_dtype(t)
            if t.ndim == 1:
                t = t.reshape(1, -1).expand(action.shape[0], -1)
        else:
            t = None
        # For add_episode, assume reward is already backprop if needed, so use same as original
        # (This method is typically used for batch loading where backprop may have been applied externally)
        reward_original = reward.clone() if self.use_torch_tensors else reward.copy()
        self._add_episode_internal(
            obs,
            action,
            reward,
            terminated,
            task=t,
            winner=winner,
            reward_original=reward_original,
        )

    def add_episodes(self, episodes):
        """Add multiple episodes (batch load).

        Args:
            episodes: List of dicts with keys obs, action, reward, terminated, task (optional), winner (optional).
        """
        for ep in episodes:
            self.add_episode(
                ep["obs"],
                ep["action"],
                ep["reward"],
                ep["terminated"],
                ep.get("task"),
                ep.get("winner"),
            )

    def _episodes_with_length_at_least(self, H):
        """Return list of (ep, start) for episodes that have at least H transitions
        and a valid start index; for each, start is in [0, T-H].
        """
        out = []
        for ep in self._episodes:
            T = ep["action"].shape[0]
            if T >= H:
                for start in range(0, T - H + 1):
                    out.append((ep, start))
        return out

    def sample(self, batch_size=None, horizon=None, return_original_reward=False):
        """Sample a batch of subsequences.

        Returns
        -------
        tuple
            (obs, action, reward, terminated, task) with shapes
            (B, H+1, obs_dim), (B, H, action_dim), (B, H, 1), (B, H, 1),
            and (B, task_dim) or None if not multitask.

        Raises
        ------
        RuntimeError
            If no episode has length >= horizon or buffer is empty.
        """
        B = batch_size if batch_size is not None else self.batch_size
        H = horizon if horizon is not None else self.horizon

        candidates = self._episodes_with_length_at_least(H)
        if len(candidates) == 0:
            raise RuntimeError(
                "TDMPC2ReplayBuffer.sample: no episode has length >= horizon. "
                "Add episodes or reduce horizon."
            )

        # Sample B (ep, start) with replacement
        inds = random.choices(range(len(candidates)), k=B)
        chunks = [candidates[i] for i in inds]

        obs_list = []
        action_list = []
        reward_list = []
        terminated_list = []
        task_list = []

        for ep, start in chunks:
            obs_list.append(ep["obs"][start : start + H + 1])
            action_list.append(ep["action"][start : start + H])

            if return_original_reward:
                reward_list.append(ep["reward_original"][start : start + H])
            else:
                reward_list.append(ep["reward"][start : start + H])
            terminated_list.append(ep["terminated"][start : start + H])
            if self.multitask and ep.get("task") is not None:
                task_list.append(ep["task"][start])

        if self.use_torch_tensors:
            obs = torch.stack(obs_list, dim=0)
            action = torch.stack(action_list, dim=0)
            reward = torch.stack(reward_list, dim=0)
            terminated = torch.stack(terminated_list, dim=0)
            if reward.dim() == 2:
                reward = reward.unsqueeze(-1)
            if terminated.dim() == 2:
                terminated = terminated.unsqueeze(-1)
            obs = obs.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            reward = reward.to(self.device, non_blocking=True)
            terminated = terminated.to(self.device, non_blocking=True)
            if task_list:
                task = torch.stack(task_list, dim=0).to(self.device, non_blocking=True)
            else:
                task = None
        else:
            obs = np.stack(obs_list, axis=0)
            action = np.stack(action_list, axis=0)
            reward = np.stack(reward_list, axis=0)
            terminated = np.stack(terminated_list, axis=0)
            if reward.ndim == 2:
                reward = reward.reshape(*reward.shape, 1)
            if terminated.ndim == 2:
                terminated = terminated.reshape(*terminated.shape, 1)
            task = np.stack(task_list, axis=0) if task_list else None

        return obs, action, reward, terminated, task

    def sample_sequences(self, batch_size, horizon):
        """Sample contiguous sequences for TD-MPC2 train. Same as sample but
        returns a 4-tuple (obs, action, reward, terminated) so it can replace
        ReplayBuffer.sample_sequences in the train loop.

        Returns
        -------
        tuple or None
            (obs_seq, action_seq, reward_seq, done_seq) with shapes
            (batch, horizon+1, obs_dim), (batch, horizon, action_dim),
            (batch, horizon, 1), (batch, horizon, 1). Returns None if no
            episode has length >= horizon.
        """
        candidates = self._episodes_with_length_at_least(horizon)
        if len(candidates) == 0:
            return None
        obs, action, reward, terminated, _ = self.sample(
            batch_size=batch_size, horizon=horizon
        )
        return obs, action, reward, terminated

    def clear(self):
        """Clear the buffer and any in-progress episode from store()."""
        self._episodes = []
        self._total_transitions = 0
        self._current_obs = []
        self._current_actions = []
        self._current_rewards = []
        self._current_dones = []
        self._current_winner = None


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer.

    Samples transitions based on their TD error priorities.
    Uses importance sampling weights to correct for the bias introduced
    by non-uniform sampling.
    """

    def __init__(
        self,
        max_size=1_000_000,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0001,
        max_beta=1.0,
        eps=1e-6,
    ):
        """Initialize Prioritized Replay Buffer.

        Parameters
        ----------
        max_size : int
            Maximum size of the buffer
        alpha : float
            Priority exponent (0 = uniform, 1 = fully prioritized)
        beta : float
            Initial importance sampling exponent
        beta_increment : float
            How much to increment beta per sample
        max_beta : float
            Maximum value for beta
        eps : float
            Small constant to ensure priorities are never zero
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = max_beta
        self.eps = eps
        self.max_priority = 1.0

        # Calculate tree capacity (must be power of 2)
        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, transition):
        """Store a transition with maximum priority."""
        idx = self.current_idx
        super().store(transition)

        # Initialize with maximum priority
        priority = (self.max_priority + self.eps) ** self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

    def sample(self, batch_size):
        """Sample a batch of transitions with priorities.

        Returns
        -------
        tuple
            (state, action, reward, next_state, done, weights, indices)
            where weights are importance sampling weights and indices
            are the buffer indices for updating priorities later.
        """
        if batch_size > self.size:
            batch_size = self.size

        if batch_size == 0:
            # Return empty batch
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                [],
            )

        # Increment beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        # Sample indices based on priorities
        indices = []
        priorities = []

        total_priority = self.sum_tree.sum(0, self.size - 1)
        # Safety check for zero priority
        if total_priority <= 0:
            total_priority = self.eps * self.size

        segment_length = total_priority / batch_size

        for i in range(batch_size):
            prefixsum = (i + random.random()) * segment_length
            idx = self.sum_tree.find_prefixsum_idx(prefixsum)
            # Clamp to valid range (shouldn't be necessary but safety check)
            idx = min(idx, self.size - 1)
            indices.append(idx)
            priorities.append(self.sum_tree[idx])

        # Compute importance sampling weights
        min_priority = self.min_tree.min(0, self.size - 1) / total_priority
        max_weight = (min_priority * self.size) ** (-self.beta)

        weights = []
        for priority in priorities:
            prob = priority / total_priority
            weight = (prob * self.size) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)

        # Get the actual transitions
        state = self.state[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_state = self.next_state[indices]
        done = self.done[indices]

        return (state, action, reward, next_state, done, weights, indices)

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions.

        Parameters
        ----------
        indices : array-like
            Buffer indices of transitions
        priorities : array-like
            New priorities (typically TD errors)
        """
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            priority_alpha = priority**self.alpha
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha

    def clear(self):
        """Clear the replay buffer, resetting it to empty state."""
        super().clear()
        # Reset trees (they will be re-initialized on next store)
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
