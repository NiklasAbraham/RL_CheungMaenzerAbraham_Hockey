import numpy as np
import random

from rl_hockey.common.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.max_size = max_size
        self.current_idx = 0
        self.size = 0

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = None

    def store(self, transition):
        state, action, reward, next_state, done = transition

        if self.size == 0:
            self.state = np.empty((self.max_size, len(state)), dtype=np.float32)
            self.action = np.empty((self.max_size, len(action)), dtype=np.float32)
            self.next_state = np.empty((self.max_size, len(state)), dtype=np.float32)
            self.reward = np.empty((self.max_size, 1), dtype=np.float32)
            self.done = np.empty((self.max_size, 1), dtype=np.float32)

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

        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
        )
    
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
    
    def __init__(self, max_size=1_000_000, alpha=0.6, beta=0.4, beta_increment=0.0001, max_beta=1.0, eps=1e-6):
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
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), [])
        
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
            priority_alpha = priority ** self.alpha
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