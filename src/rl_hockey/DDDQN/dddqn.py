import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_hockey.common.agent import Agent
from rl_hockey.DDDQN.models import DuelingDQN_Network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://arxiv.org/abs/1511.06581
class DDDQN(Agent):
    def __init__(self, state_dim, action_dim, hidden_dim=[256, 256], **user_config):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

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
        }

        self.config.update(user_config)

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
            state, action, reward, next_state, done = self.buffer.sample(
                self.config["batch_size"]
            )

            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.from_numpy(action.astype(np.int64) if action.dtype == np.float32 else action).long().to(DEVICE)
            if action.dim() > 1:
                action = action.squeeze(1)
            reward = torch.from_numpy(reward).float().to(DEVICE).squeeze(-1)

            reward_clip = self.config.get("reward_clip", None)
            if reward_clip is not None:
                reward = torch.clamp(reward, -reward_clip, reward_clip)
            
            next_state = torch.from_numpy(next_state).float().to(DEVICE)
            done = torch.from_numpy(done).float().to(DEVICE).squeeze(-1)

            with torch.no_grad():
                next_q_values = self.q_network(next_state)
                next_action = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values_target = self.q_network_target(next_state)
                next_value = next_q_values_target.gather(1, next_action).squeeze(1)

            target = reward + (1 - done) * self.config["discount"] * next_value

            current_q_values = self.q_network(state)
            current_q_value = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            if self.config.get("use_huber_loss", False):
                loss = F.smooth_l1_loss(current_q_value, target)
            else:
                loss = F.mse_loss(current_q_value, target)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            grad_clip = self.config.get("grad_clip", 10.0)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=grad_clip)
            self.optimizer.step()

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
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.config.update(checkpoint["config"])
        hidden_dim = checkpoint.get("hidden_dim", [256, 256, 256])

        self.q_network = DuelingDQN_Network(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(DEVICE)
        self.q_network_target = DuelingDQN_Network(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config["learning_rate"])

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
