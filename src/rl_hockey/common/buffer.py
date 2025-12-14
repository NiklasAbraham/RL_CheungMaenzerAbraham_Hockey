import numpy as np


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
