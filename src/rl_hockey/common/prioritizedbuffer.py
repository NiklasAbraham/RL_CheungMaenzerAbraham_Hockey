import random
import numpy as np
from rl_hockey.common.buffer import ReplayBuffer
from rl_hockey.common.sumtree import SumTree


class PERMemory(ReplayBuffer):  # stored as ( s, a, r, s_, done ) in SumTree
    def __init__(
        self,
        capacity,
        eps=0.01,
        alpha=0.6,
        beta=0.4,
        beta_increment_per_sampling=0.0001,
    ):
        super().__init__(max_size=capacity)
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def store(self, transition):
        p = self.tree.max_p
        self.tree.add(p, self.current_idx)
        super().store(transition)

    def sample(self, batch_size):
        data_indices = []
        idxs = []
        segment = self.tree.total_p() / batch_size
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            data_indices.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_p()
        is_weight = np.power(self.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        batch = (
            self.state[data_indices],
            self.action[data_indices],
            self.reward[data_indices],
            self.next_state[data_indices],
            self.done[data_indices],
        )

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
