import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_hockey.common.agent import Agent
from rl_hockey.common.buffer import PrioritizedReplayBuffer, ReplayBuffer
from rl_hockey.DDDQN.models import DuelingDQN_Network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDQN_PER(Agent):
    """Double Deep Q-Network with optional Prioritized Experience Replay.

    This implementation extends DDDQN with the ability to toggle
    Prioritized Experience Replay (PER) on or off.

    When PER is enabled:
    - Samples transitions based on TD error priorities
    - Uses importance sampling weights to correct for bias
    - Updates priorities after each training step
    """

    def __init__(
        self, state_dim, action_dim, hidden_dim=[256, 256], use_per=False, **user_config
    ):
        """Initialize DDQN with optional PER.

        Parameters
        ----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space
        hidden_dim : list
            List of hidden layer dimensions
        use_per : bool
            Whether to use Prioritized Experience Replay
        **user_config : dict
            Additional configuration parameters including:
            - batch_size: Batch size for training (default: 256)
            - learning_rate: Learning rate (default: 1e-4)
            - discount: Discount factor (default: 0.99)
            - target_update_freq: Frequency of target network updates (default: 50)
            - reward_clip: Clip rewards to this value (default: None)
            - use_huber_loss: Use Huber loss instead of MSE (default: False)
            - grad_clip: Gradient clipping value (default: 1.0)
            - eps: Initial epsilon for epsilon-greedy (default: 1.0)
            - eps_min: Minimum epsilon (default: 0.01)
            - eps_decay: Epsilon decay factor (default: 0.9995)
            - per_alpha: PER priority exponent (default: 0.6)
            - per_beta: PER initial importance sampling exponent (default: 0.4)
            - per_beta_increment: PER beta increment per sample (default: 0.0001)
            - per_max_beta: PER maximum beta value (default: 1.0)
            - per_eps: PER epsilon for priorities (default: 1e-6)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_per = use_per

        self.config = {
            "batch_size": 256,
            "learning_rate": 1e-4,
            "discount": 0.99,
            "target_update_freq": 50,
            "reward_clip": None,
            "use_huber_loss": False,
            "grad_clip": 1.0,
            "eps": 1,
            "eps_min": 0.01,
            "eps_decay": 0.9995,
            # PER parameters
            "per_alpha": 0.6,
            "per_beta": 0.4,
            "per_beta_increment": 0.0001,
            "per_max_beta": 1.0,
            "per_eps": 1e-6,
        }

        self.config.update(user_config)

        # Replace buffer with PER buffer if enabled
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                max_size=self.config.get("buffer_size", 1_000_000),
                alpha=self.config["per_alpha"],
                beta=self.config["per_beta"],
                beta_increment=self.config["per_beta_increment"],
                max_beta=self.config["per_max_beta"],
                eps=self.config["per_eps"],
            )
        else:
            self.buffer = ReplayBuffer(
                max_size=self.config.get("buffer_size", 1_000_000)
            )

        self.q_network = DuelingDQN_Network(
            state_dim, action_dim, hidden_dim=hidden_dim
        ).to(DEVICE)
        self.q_network_target = DuelingDQN_Network(
            state_dim, action_dim, hidden_dim=hidden_dim
        ).to(DEVICE)
        self.q_network_target.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.config["learning_rate"]
        )

        self.training_steps = 0

    def act(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(DEVICE)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            if deterministic:
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            else:
                eps = self.config["eps"]
                if np.random.random() > eps:
                    q_values = self.q_network(state_tensor)
                    action = q_values.argmax(dim=1).item()
                else:
                    action = np.random.randint(0, self.action_dim)

            return action

    def act_batch(self, states, deterministic=False):
        """Process a batch of states at once (for vectorized environments)"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(states).float().to(DEVICE)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            q_values = self.q_network(state_tensor)

            if deterministic:
                actions = q_values.argmax(dim=1).cpu().numpy()
            else:
                eps = self.config["eps"]
                # For each state in batch, decide whether to explore or exploit
                batch_size = state_tensor.shape[0]
                random_mask = np.random.random(batch_size) < eps

                # Get greedy actions
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()

                # Get random actions
                random_actions = np.random.randint(0, self.action_dim, size=batch_size)

                # Combine: use random where mask is True, greedy otherwise
                actions = np.where(random_mask, random_actions, greedy_actions)

            return actions

    def evaluate(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(DEVICE)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            q_values = self.q_network(state_tensor)
            max_q_value = q_values.max(dim=1)[0].item()
            return max_q_value

    def train(self, steps=1):
        losses = []

        for i in range(steps):
            # Sample from buffer (PER returns additional values)
            if self.use_per:
                state, action, reward, next_state, done, weights, indices = (
                    self.buffer.sample(self.config["batch_size"])
                )
                weights = (
                    torch.from_numpy(weights).float().to(DEVICE, non_blocking=True)
                )
            else:
                state, action, reward, next_state, done = self.buffer.sample(
                    self.config["batch_size"]
                )
                weights = None
                indices = None

            state = torch.from_numpy(state).float().to(DEVICE, non_blocking=True)
            action = (
                torch.from_numpy(
                    action.astype(np.int64) if action.dtype == np.float32 else action
                )
                .long()
                .to(DEVICE, non_blocking=True)
            )
            if action.dim() > 1:
                action = action.squeeze(1)
            reward = (
                torch.from_numpy(reward)
                .float()
                .to(DEVICE, non_blocking=True)
                .squeeze(-1)
            )

            reward_clip = self.config.get("reward_clip", None)
            if reward_clip is not None:
                reward = torch.clamp(reward, -reward_clip, reward_clip)

            next_state = (
                torch.from_numpy(next_state).float().to(DEVICE, non_blocking=True)
            )
            done = (
                torch.from_numpy(done).float().to(DEVICE, non_blocking=True).squeeze(-1)
            )

            with torch.no_grad():
                next_q_values = self.q_network(next_state)
                next_action = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values_target = self.q_network_target(next_state)
                next_value = next_q_values_target.gather(1, next_action).squeeze(1)

            target = reward + (1 - done) * self.config["discount"] * next_value

            current_q_values = self.q_network(state)
            current_q_value = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute TD errors for PER
            td_errors = (current_q_value - target).detach().cpu().numpy()

            # Compute loss with optional importance sampling weights
            if self.config.get("use_huber_loss", False):
                if self.use_per and weights is not None:
                    loss = F.smooth_l1_loss(current_q_value, target, reduction="none")
                    loss = (loss * weights).mean()
                else:
                    loss = F.smooth_l1_loss(current_q_value, target)
            else:
                if self.use_per and weights is not None:
                    loss = F.mse_loss(current_q_value, target, reduction="none")
                    loss = (loss * weights).mean()
                else:
                    loss = F.mse_loss(current_q_value, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_clip = self.config.get("grad_clip", 10.0)
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), max_norm=grad_clip
            )
            self.optimizer.step()

            # Update priorities if using PER
            if self.use_per and indices is not None:
                self.buffer.update_priorities(indices, td_errors)

            losses.append(loss.item())
            self.training_steps += 1

            if self.training_steps % self.config["target_update_freq"] == 0:
                self.q_network_target.load_state_dict(self.q_network.state_dict())

        return {"loss": losses}

    def save(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        if not filepath.endswith(".pt"):
            filepath += ".pt"

        hidden_dim = []
        for module in self.q_network.feature_network:
            if isinstance(module, torch.nn.Linear):
                hidden_dim.append(module.out_features)

        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "q_network_target": self.q_network_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": hidden_dim,
            "use_per": self.use_per,
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.config.update(checkpoint["config"])
        self.use_per = checkpoint.get("use_per", False)
        hidden_dim = checkpoint.get("hidden_dim", [256, 256, 256])

        # Reinitialize buffer with correct type
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                max_size=self.config.get("buffer_size", 1_000_000),
                alpha=self.config["per_alpha"],
                beta=self.config["per_beta"],
                beta_increment=self.config["per_beta_increment"],
                max_beta=self.config["per_max_beta"],
                eps=self.config["per_eps"],
            )
        else:
            self.buffer = ReplayBuffer(
                max_size=self.config.get("buffer_size", 1_000_000)
            )

        self.q_network = DuelingDQN_Network(
            self.state_dim, self.action_dim, hidden_dim=hidden_dim
        ).to(DEVICE)
        self.q_network_target = DuelingDQN_Network(
            self.state_dim, self.action_dim, hidden_dim=hidden_dim
        ).to(DEVICE)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.config["learning_rate"]
        )

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_network_target.load_state_dict(checkpoint["q_network_target"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            pass

        self.training_steps = checkpoint.get("training_steps", 0)

    def on_episode_start(self, episode):
        if self.config["eps_decay"] is not None:
            self.config["eps"] = max(
                self.config["eps_min"], self.config["eps"] * self.config["eps_decay"]
            )
        return {"eps": self.config["eps"]}
