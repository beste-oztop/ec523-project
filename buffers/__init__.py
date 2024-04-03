import random
import torch

class ReplayBuffer():

    def __init__(self, buffer_size,
                 single_observation_space,
                 single_action_space,
                 device,
                 handle_timeout_termination=False):
        self.buffer_size = buffer_size
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.device = device
        self.handle_timeout_termination = False
        self.buffer = []
        self.idx = 0

    def add(self, obs, real_next_obs, actions, rewards, terminations, infos):
        data = (obs, real_next_obs, actions, rewards, terminations, infos)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.idx] = data
            self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_batch, next_obs_batch, action_batch, reward_batch, termination_batch, info_batch = map(list, zip(*batch))

        # convert the data stored in buffer to tensor
        obs_batch = torch.tensor(obs_batch, device=self.device, dtype=torch.float)
        next_obs_batch = torch.tensor(next_obs_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        termination_batch = torch.tensor(termination_batch, device=self.device)
        # info_batch = torch.tensor(info_batch, device=self.device)

        return obs_batch, next_obs_batch, action_batch, reward_batch, termination_batch

