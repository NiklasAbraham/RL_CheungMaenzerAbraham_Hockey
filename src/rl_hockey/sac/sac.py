import torch

from rl_hockey.common import Agent
from rl_hockey.sac.models import Actor, Critic


class SAC(Agent):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)

    def act(self, state):
        mean, logvar = self.actor(state)
    
        # sampling (reparametrization)

        return mean

    def evaluate(self, state):
        action = self.act(state)

        Q = torch.min(self.critic1(state, action), self.critic2(state, action))

        # entropy

        return Q
    
    def train(self):
        pass
