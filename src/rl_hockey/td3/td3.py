import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from rl_hockey.common import *
from rl_hockey.td3.models import Actor, TwinCritic


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://arxiv.org/pdf/1802.09477
class TD3(Agent):
    def __init__(self, state_dim, action_dim, **user_config):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.config = {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "actor_dim": 256,
            "critic_dim": 256,
            "actor_n_layers": 1,
            "critic_n_layers": 1,
            "policy_update_delay": 2,
            "batch_size": 256,
            "discount": 0.99,
            "tau": 0.005,
            "noise_type": "normal",
            "action_min": -1.0,
            "action_max": 1.0,
            "exploration_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "target_network_update_steps": 500,
            "verbose": False,
        }
        self.config.update(user_config)
        self.actor_lr = self.config["actor_lr"]
        self.critic_lr = self.config["critic_lr"]
        self.actor_dim = self.config["actor_dim"]
        self.critic_dim = self.config["critic_dim"]
        self.actor_n_layers = self.config["actor_n_layers"]
        self.critic_n_layers = self.config["critic_n_layers"]
        self.batch_size = self.config["batch_size"]
        self.discount = self.config["discount"]
        self.tau = self.config["tau"]
        self.learn_steps_counter = 0
        self.update_actor_iter = self.config["policy_update_delay"]
        self.action_min = self.config["action_min"]
        self.action_max = self.config["action_max"]
        self.noise_type = self.config["noise_type"]  # Not implemented yet
        self.exploration_noise = self.config["exploration_noise"]
        self.policy_noise = self.config["policy_noise"]
        self.noise_clip = self.config["noise_clip"]
        self.target_network_update_steps = self.config["target_network_update_steps"]
        self.verbose = self.config["verbose"]

        self.actor = Actor(
            state_dim, action_dim, self.actor_dim, self.actor_n_layers
        ).to(DEVICE)
        self.actor_target = Actor(
            state_dim, action_dim, self.actor_dim, self.actor_n_layers
        ).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.twincritic_online = TwinCritic(
            state_dim, action_dim, self.critic_dim, self.critic_n_layers
        ).to(DEVICE)
        self.twincritic_target = TwinCritic(
            state_dim, action_dim, self.critic_dim, self.critic_n_layers
        ).to(DEVICE)
        self.twincritic_target.load_state_dict(self.twincritic_online.state_dict())

        if self.verbose:
            print("Initialized Actor:")
            print(self.actor)

            print("Initialized Critics:")
            print(self.twincritic_online.critic1)

        self.critic_optimizer = optim.Adam(
            self.twincritic_online.parameters(),
            lr=self.critic_lr,
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # TODO: implement lr scheduler
        # self.scheduler = LambdaLR(critic_optimizer, lr_lambda=)

        match self.config["noise_type"]:
            case "normal":
                self.exploration_noise_dist = torch.distributions.Normal(
                    loc=torch.zeros(self.action_dim),
                    scale=self.exploration_noise * torch.ones(self.action_dim),
                )
                self.policy_noise_dist = torch.distributions.Normal(
                    loc=torch.zeros(self.action_dim),
                    scale=self.policy_noise * torch.ones(self.action_dim),
                )
            case _:
                raise ValueError(f"Unimplemented noise type: {self.config['noise']}")

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            noise = self.exploration_noise_dist.sample((len(state),)).to(DEVICE)
            pred_action = self.actor(state)
            new_action = pred_action + noise
            new_action = torch.clamp(new_action, self.action_min, self.action_max)

            return new_action.squeeze(0).cpu().numpy()

    def evaluate(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            # 1. Get the deterministic action (No sampling, no log_prob)
            action = self.actor(state)

            # 2. Get the Q-values from both critics
            q1, q2 = self.twincritic_online(state, action)

            # 3. Take the conservative estimate (min)
            value = torch.min(q1, q2)

            return value.squeeze().item()

    def train(self, steps=1):
        critic_losses = []
        actor_losses = []

        for i in range(steps):
            state, action, reward, next_state, done = self.buffer.sample(
                self.batch_size
            )

            state = torch.from_numpy(state).to(dtype=torch.float32, device=DEVICE)
            action = torch.from_numpy(action).to(dtype=torch.float32, device=DEVICE)
            reward = torch.from_numpy(reward).to(dtype=torch.float32, device=DEVICE)
            next_state = torch.from_numpy(next_state).to(
                dtype=torch.float32, device=DEVICE
            )
            done = torch.from_numpy(done).to(dtype=torch.float32, device=DEVICE)

            # calculate critic target
            with torch.no_grad():
                next_action = self.actor_target(next_state)

                # Double clipping with Gaussian Noise
                next_action += torch.clamp(
                    self.policy_noise_dist.sample((self.batch_size,)).to(DEVICE),
                    -self.noise_clip,
                    self.noise_clip,
                )
                next_action = torch.clamp(next_action, self.action_min, self.action_max)

                # Double Q Network with minimum Q value
                q1_target, q2_target = self.twincritic_target(next_state, next_action)
                next_value = torch.min(q1_target, q2_target)

                target = reward + (1 - done) * self.discount * next_value

            q1_online, q2_online = self.twincritic_online(state, action)
            c1_loss = F.mse_loss(q1_online, target)
            c2_loss = F.mse_loss(q2_online, target)
            critic_loss = c1_loss + c2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(
                self.twincritic_online.parameters(), max_norm=0.5
            )
            self.critic_optimizer.step()

            critic_losses.append(critic_loss.item())

            self.learn_steps_counter += 1

            # Accumulative updates for actor
            if self.learn_steps_counter % self.update_actor_iter != 0:
                continue

            self.actor_optimizer.zero_grad()
            new_action = self.actor(state)
            actor_q1_loss, _ = self.twincritic_online(state, new_action)
            actor_loss = -torch.mean(actor_q1_loss)
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())

            if self.learn_steps_counter % self.target_network_update_steps:
                self.update_network_parameters()

        return {"critic_loss": critic_losses, "actor_loss": actor_losses}

    def update_network_parameters(self):
        # This prevents PyTorch from building a graph for the update
        with torch.no_grad():
            # Update Critic Target
            for online_p, target_p in zip(
                self.twincritic_online.parameters(), self.twincritic_target.parameters()
            ):
                target_p.data.copy_(
                    self.tau * online_p.data + (1 - self.tau) * target_p.data
                )

            # Update Actor Target
            for online_p, target_p in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_p.data.copy_(
                    self.tau * online_p.data + (1 - self.tau) * target_p.data
                )

    def save(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        if not filepath.endswith(".pt"):
            filepath += ".pt"

        checkpoint = {
            "actor": self.actor.state_dict(),
            "twincritic_online": self.twincritic_online.state_dict(),
            "twincritic_target": self.twincritic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)

        self.actor.load_state_dict(checkpoint["actor"])
        self.twincritic_online.load_state_dict(checkpoint["twincritic_online"])
        self.twincritic_target.load_state_dict(checkpoint["twincritic_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

    def on_episode_start(self, episode):
        pass
