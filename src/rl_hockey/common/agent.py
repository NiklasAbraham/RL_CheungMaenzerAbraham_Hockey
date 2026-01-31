import inspect
from abc import ABC, abstractmethod

from rl_hockey.common.prioritizedbuffer import PERMemory
from rl_hockey.common.buffer import ReplayBuffer


class Agent(ABC):
    def __init__(self,  deterministic=False, priority_replay: bool = False, normalize_obs: bool = False):
        self.deterministic = deterministic

        if priority_replay:
            self.buffer = PERMemory(normalize_obs=normalize_obs)
        else:
            self.buffer = ReplayBuffer(normalize_obs=normalize_obs)
    
    def store_transition(self, transition, winner=None):
        """Stores a transition in the replay buffer.
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
            winner: Optional winner information (1 for agent win, -1 for loss, 0 for draw).
                Only used by buffers that support it (e.g., TDMPC2ReplayBuffer).
        """
        # Check if buffer.store() accepts winner parameter
        store_sig = inspect.signature(self.buffer.store)
        if 'winner' in store_sig.parameters:
            self.buffer.store(transition, winner=winner)
        else:
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

    def log_architecture(self):
        """
        Logs the network architecture for the agent.
        Override this method in subclasses to provide agent-specific architecture details.
        Returns a formatted string describing the architecture.
        """
        return "Network architecture logging not implemented for this agent type."
