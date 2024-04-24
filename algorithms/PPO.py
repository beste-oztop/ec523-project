import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from buffers import ReplayBuffer
from networks import AgentPPO

class PPO():

    def __init__(self, envs, writer, args):
        super().__init__()

        # seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.envs = envs
        self.writer = writer
        self.agent = AgentPPO(envs).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)


        self.start_time = time.time()


    
    def environment_interaction(self,args):
        obs = torch.zeros((args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(self.device)

        next_obs, _ = self.envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(args.num_envs).to(self.device)

        for step in range(self.args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            action, logprob, _, value = self.agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

        return obs, actions, logprobs, rewards, dones, values
    

    def compute_advantage(self, rewards, dones, values):
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0
        last_return = 0
        for t in reversed(range(self.args.num_steps)):
            mask = 1 - dones[t].float()
            delta = rewards[t] + self.args.gamma * values[t + 1] * mask - values[t]
            advantages[t] = last_advantage = delta + self.args.gamma * self.args.gae_lambda * last_advantage * mask
            returns[t] = last_return = rewards[t] + self.args.gamma * last_return * mask
        return advantages, returns


    def flatten_batch(self, obs, logprobs, actions, advantages, returns, values):
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values
    
    
    def optimize_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []

        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)

                # Policy loss
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                # Value loss
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * torch.mean(v_loss_max)
                else:
                    v_loss = 0.5 * torch.mean((newvalue - b_returns[mb_inds]) ** 2)

                entropy_loss = torch.mean(entropy)
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break


    def execute(self,args):
        #start of the game
        global_step = 0

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            obs, actions, logprobs, rewards, dones, values = self.environment_interaction()
            advantages, returns = self.compute_advantage(rewards, dones, values)
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = self.flatten_batch(obs, logprobs, actions, advantages, returns, values)
            pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var = self.optimize_policy(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Logging and monitoring
            global_step += self.args.num_envs
            print("SPS:", int(global_step / (time.time() - self.start_time)))

            # Log metrics using SummaryWriter
            if global_step % self.args.log_interval == 0:
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
                self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)
