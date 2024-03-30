import random

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
        return obs_batch, next_obs_batch, action_batch, reward_batch, termination_batch, info_batch

