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
        # TODO for Beste
        pass

    def sample(self, batch_size):
        # TODO for Beste
        pass
