import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self, state_dim, action_dim, latent_dim=256, num_layers=1, activation=nn.ReLU
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            activation(),
        )
        for i in range(num_layers):
            self.net.append(nn.Linear(latent_dim, latent_dim))
            self.net.append(activation())

        self.net.append(nn.Linear(latent_dim, action_dim))
        self.net.append(nn.Tanh())

    def forward(self, state):
        return self.net(state.to(torch.float32))


class TwinCritic(nn.Module):
    """
    Implementation of a pair of Critic (online and target).

    Allows us to use criterion as if it were a single network.
    """

    def __init__(
        self, state_dim, action_dim, latent_dim=256, num_layers=1, activation=nn.ReLU
    ):
        super().__init__()

        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            activation(),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            activation(),
        )

        for i in range(num_layers):
            self.critic1.append(nn.Linear(latent_dim, latent_dim))
            self.critic1.append(activation())

            self.critic2.append(nn.Linear(latent_dim, latent_dim))
            self.critic2.append(activation())

        self.critic1.append(nn.Linear(latent_dim, 1))
        self.critic2.append(nn.Linear(latent_dim, 1))

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.critic1(input), self.critic2(input)
