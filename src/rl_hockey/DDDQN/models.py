import torch.nn as nn


# https://arxiv.org/abs/1511.06581
class DuelingDQN_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[256, 256]):
        super(DuelingDQN_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim
        for h in hidden_dim:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.feature_network = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(input_dim, 1))
        self.value_stream = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(input_dim, action_dim))
        self.advantage_stream = nn.Sequential(*layers)

    def forward(self, state):
        features = self.feature_network(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
