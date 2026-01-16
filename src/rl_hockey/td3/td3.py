import os
import torch
import torch.nn.functional as F

from rl_hockey.common import *
from rl_hockey.td3.models import Actor, Critic
import copy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://arxiv.org/pdf/1802.09477
class TD3(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        **user_config
    ):
        super().__init__()

        self.config = {
            "learning_rate": 3e-4,
            "max_action": 1.0,
            "discount": 0.99,
            "tau": 0.005,
            "expl_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "batch_size": 256
        }
        self.config.update(user_config)

        self.actor = Actor(state_dim, action_dim, self.config["max_action"]).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config["learning_rate"])

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config["learning_rate"])

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = self.config["max_action"]
        self.discount = self.config["discount"]
        self.tau = self.config["tau"]
        self.expl_noise = self.config["expl_noise"]
        self.policy_noise = self.config["policy_noise"]
        self.noise_clip = self.config["noise_clip"]
        self.policy_freq = self.config["policy_freq"]
        self.batch_size = self.config["batch_size"]

        self.total_it = 0

    def act(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = self.actor(state).squeeze(0).cpu().numpy()
            if not deterministic:
                action += np.random.normal(
                    0, self.max_action * self.expl_noise, size=self.action_dim
                ).clip(-self.max_action, self.max_action)

            return action
        
    def act_batch(self, states, deterministic=False):
        """Process a batch of states at once (for vectorized environments)"""
        with torch.no_grad():
            states = torch.from_numpy(states).to(DEVICE)
            batch_size = states.shape[0]
            action = self.actor(states).squeeze(0).cpu().numpy()

            if not deterministic:
                action += np.random.normal(
                    0, self.max_action * self.expl_noise, size=(batch_size, self.action_dim)
                ).clip(-self.max_action, self.max_action)

            return action


    def evaluate(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            # 1. Get the deterministic action (No sampling, no log_prob)
            action = self.actor(state)

            # 2. Get the Q-values from both critics
            q1, q2 = self.critic(state, action)

            # 3. Take the conservative estimate (min)
            value = torch.min(q1, q2)
            return value.squeeze().item()


    def train(self, steps):
        critic_losses = []
        actor_losses = []

        for step in range(steps):
            self.total_it += 1
            
            # Sample replay buffer
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

            state = torch.from_numpy(state).to(dtype=torch.float32, device=DEVICE)
            action = torch.from_numpy(action).to(dtype=torch.float32, device=DEVICE)
            reward = torch.from_numpy(reward).to(dtype=torch.float32, device=DEVICE)
            next_state = torch.from_numpy(next_state).to(
                dtype=torch.float32, device=DEVICE
            )
            done = torch.from_numpy(done).to(dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                next_action = self.actor_target(next_state)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1-done) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor losses
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                
                actor_losses.append(actor_loss.item())
            
        return {"critic_loss": critic_losses, "actor_loss": actor_losses}

    def save(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        if not filepath.endswith(".pt"):
            filepath += ".pt"

        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

    def on_episode_start(self, episode):
        pass
