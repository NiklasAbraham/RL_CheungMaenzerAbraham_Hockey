import random
import numpy as np
from rl_hockey.common.buffer import ReplayBuffer
from rl_hockey.common.sumtree import SumTree


class PERMemory(ReplayBuffer):  # stored as ( s, a, r, s_, done ) in SumTree
    def __init__(
        self,
        eps=0.01,
        alpha=0.6,
        beta=0.4,
        max_beta=1.0,
        beta_increment=0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tree = SumTree(self.max_size)
        self.max_size = self.max_size
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_beta = max_beta
        self.beta_increment = beta_increment

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

        self.beta = np.min([1.0, self.beta + self.beta_increment])

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
