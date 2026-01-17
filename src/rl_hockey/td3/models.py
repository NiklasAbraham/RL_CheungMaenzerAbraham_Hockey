import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=[256, 256], activation=nn.ReLU, max_action=1.0):
        super(Actor, self).__init__()

        self.net = nn.Sequential()
        dim = state_dim
        for ld in latent_dim:
            self.net.append(nn.Linear(dim, ld))
            self.net.append(activation())
            dim = ld
        self.net.append(nn.Linear(dim, action_dim))

        self.max_action = max_action

    def forward(self, state):
        a = self.net(state)
        return self.max_action * torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=[256, 256], activation=nn.ReLU):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1_net = nn.Sequential()
        dim = state_dim + action_dim
        for ld in latent_dim:
            self.q1_net.append(nn.Linear(dim, ld))
            self.q1_net.append(activation())
            dim = ld
        self.q1_net.append(nn.Linear(dim, 1))

        # Q2 architecture
        self.q2_net = nn.Sequential()
        dim = state_dim + action_dim
        for ld in latent_dim:
            self.q2_net.append(nn.Linear(dim, ld))
            self.q2_net.append(activation())
            dim = ld
        self.q2_net.append(nn.Linear(dim, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.q1_net(sa)
        return q1