import random

import numpy as np
import torch

from rl_hockey.common.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    def __init__(
        self,
        max_size=1_000_000,
        use_torch_tensors=False,
        pin_memory=False,
        device="cpu",
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

        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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
            return (
                self.state[idx],
                self.action[idx],
                self.reward[idx],
                self.next_state[idx],
                self.done[idx],
            )

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
