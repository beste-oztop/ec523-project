import torch
from torch.distributions.normal import Normal
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


# Agent network implementation for PPO
class PPONetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            self.initalize_layer(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.initalize_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.initalize_layer(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self.initalize_layer(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.initalize_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.initalize_layer(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def initalize_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
