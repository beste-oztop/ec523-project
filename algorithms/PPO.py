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
from networks import PPONetwork


class PPO():

    def __init__(self, envs, writer, args):
        super().__init__()

        # seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.envs = envs
        self.writer = writer
        self.agent = PPONetwork(envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate)

        self.start_time = time.time()

    def environment_interaction(self, obs, actions, logprobs, rewards, dones, values, next_obs, next_done, args):

        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
            next_value = self.agent.get_value(next_obs).reshape(1, -1)

        return obs, actions, logprobs, rewards, dones, values, next_obs, next_done, next_value, infos

    def compute_advantage(self, rewards, dones, values, next_obs, next_done, next_value, args):
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def flatten_batch(self, obs, logprobs, actions, advantages, returns, values):
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def optimize_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, args):
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)

                # Policy loss
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * torch.mean(v_loss_max)
                else:
                    v_loss = 0.5 * torch.mean((newvalue - b_returns[mb_inds]) ** 2)

                entropy_loss = torch.mean(entropy)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                self.optimizer.step()
                loss = loss.detach()

                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs

    def execute(self, args):
        # start of the game
        global_step = 0
        obs = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(self.device)

        next_obs, _ = self.envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(args.num_envs).to(self.device)

        print("num_iterations:", args.num_iterations)
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            obs, actions, logprobs, rewards, dones, values, next_obs, next_done, next_value, infos = self.environment_interaction(
                obs, actions, logprobs, rewards, dones, values, next_obs, next_done, args)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            advantages, returns = self.compute_advantage(rewards, dones, values, next_obs, next_done, next_value, args)
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = self.flatten_batch(obs, logprobs, actions,
                                                                                                 advantages, returns,
                                                                                                 values)
            pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = self.optimize_policy(b_obs, b_logprobs,
                                                                                                      b_actions,
                                                                                                      b_advantages,
                                                                                                      b_returns,
                                                                                                      b_values, args)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Logging and monitoring
            global_step += args.num_envs
            print("SPS:", int(global_step / (time.time() - self.start_time)))

            # Log metrics using SummaryWriter
            if global_step % args.log_interval == 0:
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
                self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)
