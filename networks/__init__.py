import torch.nn as nn
import numpy as np

class MiniQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = np.array(env.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 20).float(),
            nn.ReLU(),
            nn.Linear(20, 10).float(),
            nn.ReLU(),
            nn.Linear(10, env.single_action_space.n).float(),
        )

    def forward(self, x):
        # Convert input tensor to float
        x = x.float()
        # Reshape input tensor to match the expected shape of the first linear layer
        x = x.view(-1, self.input_shape)
        return self.network(x)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = np.array(env.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120).float(),
            nn.ReLU(),
            nn.Linear(120, 84).float(),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n).float(),
        )

    def forward(self, x):
        # Convert input tensor to float
        x = x.float()
        # Reshape input tensor to match the expected shape of the first linear layer
        x = x.view(-1, self.input_shape)
        return self.network(x)



class DeeperQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = np.array(env.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120).float(),
            nn.ReLU(),
            nn.Linear(120, 42).float(),
            nn.ReLU(),
            nn.Linear(42, 42).float(),
            nn.ReLU(),
            nn.Linear(42, env.single_action_space.n).float(),
        )

    def forward(self, x):
        # Convert input tensor to float
        x = x.float()
        # Reshape input tensor to match the expected shape of the first linear layer
        x = x.view(-1, self.input_shape)
        return self.network(x)


class WiderQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = np.array(env.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120).float(),
            nn.ReLU(),
            nn.Linear(120, 168).float(),
            nn.ReLU(),
            nn.Linear(168, env.single_action_space.n).float(),
        )

    def forward(self, x):
        # Convert input tensor to float
        x = x.float()
        # Reshape input tensor to match the expected shape of the first linear layer
        x = x.view(-1, self.input_shape)
        return self.network(x)


