
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffers import ReplayBuffer

from networks import QNetwork



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN():

    def __init__(self, envs, writer, args):

        # seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.q_network = QNetwork(envs).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.target_network = QNetwork(envs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.envs = envs

        self.writer = writer

        self.replay_buffer = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )




    def get_epsilon_greedy_actions(self, epsilon, obs, device):
        if random.random() < epsilon:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            q_values = self.q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions


    def train_networks(self, global_step, args):
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = self.replay_buffer.sample(args.batch_size)
                self.optimize_QNetwork(data, global_step, args)

            if global_step % args.target_network_frequency == 0:
                self.update_target_network(args)

    def optimize_QNetwork(self, data, global_step, args):
        with torch.no_grad():
            # print(data)
            obs_batch, next_obs_batch, action_batch, reward_batch, termination_batch, info_batch = data
            target_max, _ = self.target_network(next_obs_batch).max(dim=1)
            td_target = reward_batch.flatten() + args.gamma * target_max * (1 - termination_batch.flatten())
        old_val = self.q_network(obs_batch).gather(1, action_batch).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # log losses and q_values to the writer
        if global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss", loss, global_step)
            self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self, args):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
            )


    def execute(self, args):
        obs, _ = self.envs.reset(seed=args.seed)
        for global_step in range(args.total_timesteps):
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

            actions = self.get_epsilon_greedy_actions(epsilon, obs, self.device)
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # logging to writer
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            self.replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)
            self.train_networks(global_step, args)

