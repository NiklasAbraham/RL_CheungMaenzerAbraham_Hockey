from abc import ABC, abstractmethod

from rl_hockey.common.buffer import ReplayBuffer


class Agent(ABC):
    def __init__(self):
        self.buffer = ReplayBuffer()

    def store_transition(self, transition):
        """Stores a transition in the replay buffer."""
        self.buffer.store(transition)

    @abstractmethod
    def act(self, state, deterministic=False):
        """Returns the action for a given state according to the current policy."""
        pass

    @abstractmethod
    def evaluate(self, state):
        """Returns the value of a given state."""
        pass

    @abstractmethod
    def train(self):
        """Performs a single train step."""
        pass
