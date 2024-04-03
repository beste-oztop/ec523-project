import torch.nn as nn
import numpy as np

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
