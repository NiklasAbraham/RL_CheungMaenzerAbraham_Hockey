import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_hockey.common import *
from rl_hockey.sac.models import Actor, Critic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://arxiv.org/pdf/1902.05605
class SAC(Agent):
    def __init__(self, state_dim, action_dim, **user_config):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.config = {
            "batch_size": 256,
            "learning_rate": 3e-4,
            "discount": 0.99,
            "alpha": 0.2,
            "learn_alpha": True,
            "noise": "normal",
            "max_episode_steps": 1000,
        }
        self.config.update(user_config)

        self.batch_size = self.config["batch_size"]
        self.learning_rate = self.config["learning_rate"]
        self.discount = self.config["discount"]
        self.tau = self.config["tau"]

        self.actor = Actor(state_dim, action_dim).to(DEVICE)

        self.critic1 = Critic(state_dim, action_dim, [2048, 2048]).to(DEVICE)
        self.critic2 = Critic(state_dim, action_dim, [2048, 2048]).to(DEVICE)

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.learning_rate,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )

        if self.config["learn_alpha"]:
            self.target_entropy = -action_dim

            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.config["alpha"]

        match self.config["noise"]:
            case "normal":
                self.noise_dist = noise.NormalNoise(self.action_dim)
            case "pink":
                self.noise_dist = noise.PinkNoise(
                    self.action_dim, self.config["max_episode_steps"]
                )
            case _:
                raise ValueError(f"Unknown noise type: {self.config['noise']}")

    def act(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            if deterministic:
                noise = torch.zeros((1, self.action_dim), device=DEVICE)
            else:
                noise = self.noise_dist.sample().unsqueeze(0).to(DEVICE)

            action, _ = self.actor.sample(state, noise=noise, calc_log_prob=False)

            return action.squeeze(0).cpu().numpy()

    def evaluate(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            action, log_prob = self.actor.sample(state)

            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            value = torch.min(q1, q2) - self.alpha * log_prob

            return value.squeeze().item()

    def train(self, steps=1):
        critic_losses = []
        actor_losses = []
        grad_norm_critic = []
        grad_norm_actor = []

        for i in range(steps):
            state, action, reward, next_state, done = self.buffer.sample(
                self.batch_size
            )

            state = torch.from_numpy(state).to(DEVICE)
            action = torch.from_numpy(action).to(DEVICE)
            reward = torch.from_numpy(reward).to(DEVICE)
            next_state = torch.from_numpy(next_state).to(DEVICE)
            done = torch.from_numpy(done).to(DEVICE)

            # calculate critic target
            # TODO use BN train here
            next_action, next_log_prob = self.actor.sample(next_state)

            q1 = self.critic1(
                torch.cat([state, next_state], dim=0),
                torch.cat([action, next_action], dim=0),
            )
            q1, next_q1 = torch.chunk(q1, 2, dim=0)

            q2 = self.critic2(
                torch.cat([state, next_state], dim=0),
                torch.cat([action, next_action], dim=0),
            )
            q2, next_q2 = torch.chunk(q2, 2, dim=0)

            next_value = torch.min(next_q1, next_q2) - self.alpha * next_log_prob

            target = reward + (1 - done) * self.discount * next_value
            target = target.detach()

            # update critic parameters
            c1_loss = F.mse_loss(q1, target)
            c2_loss = F.mse_loss(q2, target)
            critic_loss = c1_loss + c2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            # Compute gradient norm before step
            grad_norm_critic_val = compute_grad_norm(
                list(self.critic1.parameters()) + list(self.critic2.parameters())
            )

            self.critic_optimizer.step()

            critic_losses.append(critic_loss.item())
            grad_norm_critic.append(grad_norm_critic_val.item())

            # update actor parameters
            # TODO use BN eval here
            current_action, log_prob = self.actor.sample(state)

            q1 = self.critic1(state, current_action)
            q2 = self.critic2(state, current_action)
            actor_loss = self.alpha * log_prob - torch.min(q1, q2)
            actor_loss = actor_loss.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # Compute gradient norm before step
            grad_norm_actor_val = compute_grad_norm(self.actor.parameters())

            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            grad_norm_actor.append(grad_norm_actor_val.item())

            # update temperature
            if self.config["learn_alpha"]:
                alpha_loss = -self.log_alpha * (log_prob.detach() + self.target_entropy)
                alpha_loss = alpha_loss.mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # update critic target parameters
            for p, pt in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

            for p, pt in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            "critic_loss": critic_losses,
            "actor_loss": actor_losses,
            "grad_norm_critic": grad_norm_critic,
            "grad_norm_actor": grad_norm_actor,
        }

    def save(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        if not filepath.endswith(".pt"):
            filepath += ".pt"

        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.config["learn_alpha"] else None,
            "alpha_optimizer": self.alpha_optimizer.state_dict()
            if self.config["learn_alpha"]
            else None,
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        if self.config["learn_alpha"]:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            self.alpha = self.log_alpha.exp()

    def on_episode_start(self, episode):
        if self.config["noise"] == "pink":
            self.noise_dist.reset()
