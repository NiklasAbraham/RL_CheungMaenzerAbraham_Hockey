import logging
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

import rl_hockey.common.noise as noise
from rl_hockey.common.agent import Agent
from rl_hockey.common.utils import compute_grad_norm
from rl_hockey.sac.models import Actor, Critic

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://arxiv.org/pdf/1812.05905
class SAC(Agent):
    def __init__(self, state_dim, action_dim, deterministic=False, **user_config):
        super().__init__(deterministic=deterministic)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.config = {
            "batch_size": 256,
            "learning_rate": 3e-4,
            "discount": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learn_alpha": True,
            "noise": "normal",
            "max_episode_steps": 250,
        }
        self.config.update(user_config)

        self.batch_size = self.config["batch_size"]
        self.learning_rate = self.config["learning_rate"]
        self.discount = self.config["discount"]
        self.tau = self.config["tau"]

        self.actor = Actor(state_dim, action_dim).to(DEVICE)

        self.critic1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic1_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

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

    def act(self, state, deterministic=False, t0=None, **kwargs):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            if deterministic or self.deterministic:
                noise = torch.zeros((1, self.action_dim), device=DEVICE)
            else:
                noise = self.noise_dist.sample().unsqueeze(0).to(DEVICE)

            action, _ = self.actor.sample(state, noise=noise, calc_log_prob=False)

            return action.squeeze(0).cpu().numpy()

    def act_batch(self, states, deterministic=False, t0s=None, **kwargs):
        """Process a batch of states at once (for vectorized environments)"""
        with torch.no_grad():
            states = torch.from_numpy(states).to(DEVICE)
            batch_size = states.shape[0]

            if deterministic or self.deterministic:
                noise = torch.zeros((batch_size, self.action_dim), device=DEVICE)
            else:
                # Sample noise for each state in batch
                noise = torch.stack(
                    [self.noise_dist.sample().to(DEVICE) for _ in range(batch_size)]
                )

            actions, _ = self.actor.sample(states, noise=noise, calc_log_prob=False)

            return actions.cpu().numpy()

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

            # DEBUG: Log shapes and types from buffer
            # logger.info(f"[SAC TRAIN STEP {i}] After buffer.sample():")
            # logger.info(
            #     f"  state: type={type(state)}, shape={state.shape if hasattr(state, 'shape') else 'N/A'}, dtype={getattr(state, 'dtype', 'N/A')}"
            # )
            # logger.info(
            #     f"  action: type={type(action)}, shape={action.shape if hasattr(action, 'shape') else 'N/A'}, dtype={getattr(action, 'dtype', 'N/A')}"
            # )
            # logger.info(
            #     f"  reward: type={type(reward)}, shape={reward.shape if hasattr(reward, 'shape') else 'N/A'}, dtype={getattr(reward, 'dtype', 'N/A')}"
            # )
            # logger.info(
            #     f"    reward min={reward.min() if hasattr(reward, 'min') else 'N/A'}, max={reward.max() if hasattr(reward, 'max') else 'N/A'}, mean={reward.mean() if hasattr(reward, 'mean') else 'N/A'}"
            # )
            # if hasattr(reward, "__getitem__"):
            #     logger.info(
            #         f"    reward[0:5]={reward[:5] if len(reward) >= 5 else reward}"
            #     )
            # logger.info(
            #     f"  next_state: type={type(next_state)}, shape={next_state.shape if hasattr(next_state, 'shape') else 'N/A'}, dtype={getattr(next_state, 'dtype', 'N/A')}"
            # )
            # logger.info(
            #     f"  done: type={type(done)}, shape={done.shape if hasattr(done, 'shape') else 'N/A'}, dtype={getattr(done, 'dtype', 'N/A')}"
            # )

            state = torch.from_numpy(state).to(DEVICE)
            action = torch.from_numpy(action).to(DEVICE)
            reward = torch.from_numpy(reward).to(DEVICE)
            next_state = torch.from_numpy(next_state).to(DEVICE)
            done = torch.from_numpy(done).to(DEVICE)

            # DEBUG: Log shapes after conversion to torch
            # logger.info(f"[SAC TRAIN STEP {i}] After torch.from_numpy():")
            # logger.info(
            #     f"  state: shape={state.shape}, dtype={state.dtype}, device={state.device}"
            # )
            # logger.info(
            #     f"  action: shape={action.shape}, dtype={action.dtype}, device={action.device}"
            # )
            # logger.info(
            #     f"  reward: shape={reward.shape}, dtype={reward.dtype}, device={reward.device}"
            # )
            # logger.info(
            #     f"    reward min={reward.min().item():.6f}, max={reward.max().item():.6f}, mean={reward.mean().item():.6f}"
            # )
            # logger.info(
            #     f"    reward[0:5]={reward[:5].squeeze().tolist() if reward.numel() >= 5 else reward.squeeze().tolist()}"
            # )
            # logger.info(
            #     f"  next_state: shape={next_state.shape}, dtype={next_state.dtype}, device={next_state.device}"
            # )
            # logger.info(
            #     f"  done: shape={done.shape}, dtype={done.dtype}, device={done.device}"
            # )

            # calculate critic target
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state)

                # DEBUG: Log shapes in target calculation
                # logger.info(f"[SAC TRAIN STEP {i}] Target calculation:")
                # logger.info(
                #     f"  next_action: shape={next_action.shape}, dtype={next_action.dtype}"
                # )
                # logger.info(
                #     f"  next_log_prob: shape={next_log_prob.shape}, dtype={next_log_prob.dtype}"
                # )

                q1 = self.critic1_target(next_state, next_action)
                q2 = self.critic2_target(next_state, next_action)
                # logger.info(
                #     f"  q1: shape={q1.shape}, dtype={q1.dtype}, min={q1.min().item():.6f}, max={q1.max().item():.6f}"
                # )
                # logger.info(
                #     f"  q2: shape={q2.shape}, dtype={q2.dtype}, min={q2.min().item():.6f}, max={q2.max().item():.6f}"
                # )

                next_value = torch.min(q1, q2) - self.alpha * next_log_prob
                # logger.info(
                #     f"  next_value: shape={next_value.shape}, dtype={next_value.dtype}, min={next_value.min().item():.6f}, max={next_value.max().item():.6f}"
                # )

                target = reward + (1 - done) * self.discount * next_value
                # logger.info(
                #     f"  target: shape={target.shape}, dtype={target.dtype}, min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}"
                # )
                # logger.info(
                #     f"    target[0:5]={target[:5].squeeze().tolist() if target.numel() >= 5 else target.squeeze().tolist()}"
                # )
                # logger.info(
                #     f"    reward component: min={reward.min().item():.6f}, max={reward.max().item():.6f}, mean={reward.mean().item():.6f}"
                # )
                # logger.info(
                #     f"    (1-done) component: min={(1 - done).min().item():.6f}, max={(1 - done).max().item():.6f}, mean={(1 - done).mean().item():.6f}"
                # )
                # logger.info(
                #     f"    discount={self.discount}, next_value component: min={(self.discount * next_value).min().item():.6f}, max={(self.discount * next_value).max().item():.6f}"
                # )

            # update critic parameters
            c1_loss = F.mse_loss(self.critic1(state, action), target)
            c2_loss = F.mse_loss(self.critic2(state, action), target)
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
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "config": self.config,
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)

        # Load state_dim, action_dim, and config if available (for compatibility)
        if "state_dim" in checkpoint:
            self.state_dim = checkpoint["state_dim"]
        if "action_dim" in checkpoint:
            self.action_dim = checkpoint["action_dim"]
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        if self.config["learn_alpha"]:
            if checkpoint.get("log_alpha") is not None:
                self.log_alpha = checkpoint["log_alpha"].to(DEVICE)
                if checkpoint.get("alpha_optimizer") is not None:
                    self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
                self.alpha = self.log_alpha.exp()
            else:
                # Old checkpoint without log_alpha, initialize it
                self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
                self.alpha_optimizer = optim.Adam(
                    [self.log_alpha], lr=self.learning_rate
                )
                self.alpha = self.log_alpha.exp()

    def on_episode_start(self, episode):
        if self.config["noise"] == "pink":
            self.noise_dist.reset()
