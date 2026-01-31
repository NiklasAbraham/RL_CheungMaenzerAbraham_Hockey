"""Decoy Policy agent for mimicking agent behavior."""

import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from rl_hockey.common.agent import Agent
from rl_hockey.Decoy_Policy.decoy_network import DecoyPolicyNetwork

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoyPolicy(Agent):
    """
    Decoy policy agent that learns to mimic another agent's behavior.

    This agent learns through behavioral cloning - it trains a policy network
    to predict the actions taken by a target agent in various states.
    """

    def __init__(
        self,
        obs_dim=18,
        action_dim=4,
        hidden_layers=[256, 256],
        learning_rate=3e-4,
        priority_replay=False,
        normalize_obs=False,
        buffer_max_size=100_000,
    ):
        super().__init__(priority_replay=priority_replay, normalize_obs=normalize_obs)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.network = DecoyPolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.buffer.max_size = buffer_max_size

        self.training_step = 0

    def act(self, state, deterministic=False, t0=None, **kwargs):
        """Returns the action for a given state according to the current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            if deterministic:
                action = self.network.mean_action(state_tensor)
            else:
                action, _ = self.network.sample(state_tensor)
            return action.cpu().numpy().flatten()

    def evaluate(self, state):
        """
        Returns a placeholder value for the state.
        Decoy policies don't have value estimation.
        """
        return 0.0

    def train(self, steps=1):
        """
        Performs behavioral cloning training steps.

        Trains the policy to predict the actions from the collected demonstrations.
        Uses mean squared error between predicted and actual actions.

        Args:
            steps: Number of gradient steps to perform

        Returns:
            Dictionary with training metrics
        """
        if self.buffer.size < 256:
            return {"loss": 0.0, "n_samples": self.buffer.size}

        metrics = {
            "loss": 0.0,
            "mse_loss": 0.0,
            "n_samples": self.buffer.size,
        }
        valid_steps = 0

        for _ in range(steps):
            batch_size = min(256, self.buffer.size)
            state, action, reward, next_state, done = self.buffer.sample(batch_size)

            states = torch.FloatTensor(
                state.cpu().numpy() if hasattr(state, "cpu") else state
            ).to(DEVICE)
            actions_target = torch.FloatTensor(
                action.cpu().numpy() if hasattr(action, "cpu") else action
            ).to(DEVICE)

            actions_pred = self.network.mean_action(states)

            mse_loss = nn.functional.mse_loss(actions_pred, actions_target)
            loss = mse_loss

            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.training_step += 1
            valid_steps += 1
            metrics["loss"] += loss.item()
            metrics["mse_loss"] += mse_loss.item()

        if valid_steps > 0:
            metrics["loss"] /= valid_steps
            metrics["mse_loss"] /= valid_steps
        metrics["training_step"] = self.training_step

        return metrics

    def save(self, filepath):
        """Saves the decoy policy to the specified filepath."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "training_step": self.training_step,
            "buffer_size": self.buffer.size,
        }

        torch.save(save_dict, filepath)
        logger.info(f"Saved DecoyPolicy to {filepath}")

    def load(self, filepath):
        """Loads the decoy policy from the specified filepath."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=DEVICE)

        self.obs_dim = checkpoint.get("obs_dim", self.obs_dim)
        self.action_dim = checkpoint.get("action_dim", self.action_dim)
        self.hidden_layers = checkpoint.get("hidden_layers", self.hidden_layers)
        self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
        self.training_step = checkpoint.get("training_step", 0)

        self.network = DecoyPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=self.hidden_layers,
        ).to(DEVICE)

        self.network.load_state_dict(checkpoint["network_state_dict"])

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            f"Loaded DecoyPolicy from {filepath} (training_step={self.training_step})"
        )

    def log_architecture(self):
        """Logs the network architecture for the decoy policy."""
        architecture_str = f"\n{'='*80}\n"
        architecture_str += "DECOY POLICY ARCHITECTURE\n"
        architecture_str += f"{'='*80}\n"
        architecture_str += f"Observation Dimension: {self.obs_dim}\n"
        architecture_str += f"Action Dimension: {self.action_dim}\n"
        architecture_str += f"Hidden Layers: {self.hidden_layers}\n"
        architecture_str += f"Learning Rate: {self.learning_rate}\n"
        architecture_str += f"Training Step: {self.training_step}\n"
        architecture_str += f"\nNetwork:\n{self.network}\n"
        architecture_str += f"{'='*80}\n"
        return architecture_str
