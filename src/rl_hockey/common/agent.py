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
    def train(self, steps=1):
        """Performs `steps` gradient steps."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Saves the agent's model to the specified filepath."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Loads the agent's model from the specified filepath."""
        pass

    def on_episode_start(self, episode):
        """Hook that is called at the start of each episode."""
        pass

    def on_episode_end(self, episode):
        """Hook that is called at the end of each episode."""
        pass
