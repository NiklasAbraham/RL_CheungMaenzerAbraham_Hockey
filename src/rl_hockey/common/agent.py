from abc import ABC, abstractmethod

from rl_hockey.common.buffer import Buffer


class Agent(ABC):
    def __init__(self):
        self.buffer = Buffer()

    def store_transaction(self, transaction):
        """Stores a transaction in the replay buffer."""
        self.buffer.store(transaction)

    @abstractmethod
    def act(self, state):
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
