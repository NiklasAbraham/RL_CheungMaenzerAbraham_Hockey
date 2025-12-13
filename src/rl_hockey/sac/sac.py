import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from rl_hockey.common import Agent
from rl_hockey.sac.models import Actor, Critic


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC(Agent):
    def __init__(self, state_dim, action_dim, batch_size=256, learning_rate=3e-4, discount=0.99, tau=0.005, alpha=0.2, learn_alpha=True):
        super().__init__()

        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.learn_alpha = learn_alpha

        self.actor = Actor(state_dim, action_dim).to(DEVICE)

        self.critic1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic1_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        if self.learn_alpha:
            self.target_entropy = -action_dim

            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def act(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            action, _ = self.actor.sample(state, deterministic=deterministic, calc_log_prob=False)

            return action.squeeze(0).cpu().numpy()

    def evaluate(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            action, log_prob = self.actor.sample(state)

            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            value = torch.min(q1, q2) - self.alpha*log_prob

            return value.squeeze().item()

    def train(self, steps=1):
        critic_losses = []
        actor_losses = []

        for i in range(steps):
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

            state = torch.from_numpy(state).to(DEVICE)
            action = torch.from_numpy(action).to(DEVICE)
            reward = torch.from_numpy(reward).to(DEVICE)
            next_state = torch.from_numpy(next_state).to(DEVICE)
            done = torch.from_numpy(done).to(DEVICE)

            # calculate critic target
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state)
                
                q1 = self.critic1_target(next_state, next_action)
                q2 = self.critic2_target(next_state, next_action)
                next_value = torch.min(q1, q2) - self.alpha*next_log_prob

                target = reward + (1 - done)*self.discount*next_value

            # update critic parameters
            c1_loss = F.mse_loss(self.critic1(state, action), target)
            c2_loss = F.mse_loss(self.critic2(state, action), target)
            critic_loss = c1_loss + c2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            critic_losses.append(critic_loss.item())

            # update actor parameters
            current_action, log_prob = self.actor.sample(state)
            
            q1 = self.critic1(state, current_action)
            q2 = self.critic2(state, current_action)
            actor_loss = self.alpha*log_prob - torch.min(q1, q2)
            actor_loss = actor_loss.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())

            # update temperature
            if self.learn_alpha:
                alpha_loss = -self.log_alpha * (log_prob.detach() + self.target_entropy)
                alpha_loss = alpha_loss.mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # update critic target parameters
            for p, pt in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                pt.data.copy_(self.tau*p.data + (1 - self.tau)*pt.data)

            for p, pt in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                pt.data.copy_(self.tau*p.data + (1 - self.tau)*pt.data)

        return {'critic_loss': critic_losses, 'actor_loss': actor_losses}
