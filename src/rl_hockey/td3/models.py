import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = nn.functional.relu(self.l1(state))
        a = nn.functional.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = nn.functional.relu(self.q1_l1(sa))
        q1 = nn.functional.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = nn.functional.relu(self.q2_l1(sa))
        q2 = nn.functional.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = nn.functional.relu(self.q1_l1(sa))
        q1 = nn.functional.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        return q1