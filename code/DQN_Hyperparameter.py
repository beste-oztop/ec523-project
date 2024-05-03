import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Virtual_BHD3_DQN"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "SpaceInvadersNoFrameskip-v4" #"BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    epsilon: float = 0.1
    """epsilon for eps-greedy"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # Model Selection Algorithm Parameters
    hparam_to_tune: str = "epsilon"
    # num_base_learners: int = 1
    # base_learners_hparam = [0.1]    # HF prefered param
    num_base_learners: int = 9
    base_learners_hparam = [1, 0.7, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01]



class BalancingHyperparamDoublingDataDriven:
    def __init__(self, m, dmin, delta =0.01, 
        c = 1, classic = True, empirical= False ):
        
        ### balancing_test_multiplier = c
        ### initial_putative_bound = dmin

        self.empirical = empirical

        self.minimum_putative = .0001
        self.maximum_putative = 10000

        self.classic = classic ### This corresponds to classic sampling among the algorithms

        self.m = m
        
        self.dmin = max(dmin, self.minimum_putative) ### this is dmin
        self.putative_bounds_multipliers = [dmin for _ in range(m)]
        

        self.balancing_potentials = [dmin*np.sqrt(1) for _ in range(m)]


        ### check these putative bounds are going up


        self.c = c ### This is the Hoeffding constant
        self.T = 1
        self.delta = delta
        

        self.all_rewards = 0

        ### these store the optimistic and pessimistic estimators of Vstar for all 
        ### base algorithms.


        self.cumulative_rewards = [0 for _ in range(self.m)]
        self.mean_rewards = [0 for _ in range(self.m)]

        self.num_plays = [0 for _ in range(self.m)]

        #self.vstar_lowerbounds = [-float("inf") for _ in range(self.m)]
        #self.vstar_upperbounds = [float("inf") for _ in range(self.m)]


        self.normalize_distribution()
        


    def sample_base_index(self):
        if self.classic:
            return np.argmin(self.balancing_potentials)
        else:
            if sum([np.isnan(x) for x in self.base_probas]) > 0:
                raise ValueError("Found Nan Values in the sampling procedure for base index")
                
                #IPython.embed()
            sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
            return sample_array[0]


    def normalize_distribution(self):
        if self.classic:
            self.base_probas = [0 for _ in range(self.m)]
            self.base_probas[self.sample_base_index()] = 1 

        else:
            #raise ValueError("Not implemented randomized selection rule for the algorithm index. Implement.")
            distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]

            normalization_factor = np.sum(distribution_base_parameters)
            self.base_probas = [x/normalization_factor for x in distribution_base_parameters]
    


    def get_distribution(self):
        return self.base_probas



    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        self.all_rewards += reward

        self.cumulative_rewards[algo_idx] += reward
        self.num_plays[algo_idx] += 1

        #### Update average reward per algorithm so far. 
        self.mean_rewards[algo_idx] = self.cumulative_rewards[algo_idx]*1.0/self.num_plays[algo_idx]


        U_t_lower_bounds = [0 for _ in range(self.m)]
        hoeffding_bonuses = [ self.c*np.sqrt(self.num_plays[i]*np.log((self.num_plays[i]+1)*1.0/self.delta)) for i in range(self.m)]
        #hoeffding_bonuses = [ self.c*np.sqrt(self.num_plays[i]) for i in range(self.m)]


        for i in range(self.m):
            U_t_lower_bounds[i] = (self.cumulative_rewards[i] - hoeffding_bonuses[i])*1.0/np.sqrt(max(self.num_plays[i], 1))


        #U_i_t_upper_bound = (self.cumulative_rewards[algo_idx] - hoeffding_bonuses[algo_idx])*1.0/np.sqrt(self.num_plays[algo_idx])
        U_i_t_upper_bound = (self.cumulative_rewards[algo_idx] + hoeffding_bonuses[algo_idx])*1.0/np.sqrt(self.num_plays[algo_idx])


        empirical_regret_estimator = self.num_plays[algo_idx]*( max(U_t_lower_bounds) - U_i_t_upper_bound )


        
        if self.empirical:
            clipped_regret = min( empirical_regret_estimator,  2*self.balancing_potentials[algo_idx])
            self.balancing_potentials[algo_idx] = max(clipped_regret, self.balancing_potentials[algo_idx], self.dmin*np.sqrt(self.num_plays[algo_idx]) )
            ### Compute implied putative bound multipliers.
            self.putative_bounds_multipliers[algo_idx] = max(self.balancing_potentials[algo_idx]/np.sqrt(self.num_plays[algo_idx]), self.dmin)



        else:
            ### test for misspecification
            if empirical_regret_estimator > self.putative_bounds_multipliers[algo_idx]*np.sqrt(self.num_plays[algo_idx]):
                self.putative_bounds_multipliers[algo_idx]= min(2*self.putative_bounds_multipliers[algo_idx], self.maximum_putative)


            self.balancing_potentials[algo_idx] = self.putative_bounds_multipliers[algo_idx]*np.sqrt(self.num_plays[algo_idx])





        print("Curr reward ", reward)
        print("All rewards ", self.all_rewards)
        print("Cumulative rewards ", self.cumulative_rewards)
        print("Num plays ", self.num_plays)
        print("Mean rewards ", self.mean_rewards)
        #print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        self.T += 1



        self.normalize_distribution()




def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def normalize_episodic_return(episodic_return, normalizer_const):
    if episodic_return < normalizer_const:
        normalized_return = episodic_return/normalizer_const
    else:
        normalized_return = 1
    return normalized_return


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class VirtualBaseLearner():

    def __init__(self, base_index, args):
        super().__init__()
        self.base_index = base_index
    

    def set_base_learner(self, base_index, optimizer, args):
        if args.hparam_to_tune == "learning_rate":
            for g in optimizer.param_groups:
                g['lr'] = args.base_learners_hparam[self.base_index]
        elif args.hparam_to_tune == "epsilon":
            args.epsilon = args.base_learners_hparam[self.base_index]
        else:
            raise NotImplementedError


    def get_hparam(self):
        return args.base_learners_hparam[self.base_index]



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    # base learners initiation
    m = args.num_base_learners
    base_learners = []
    for i in range(m):
        virtual_learner = VirtualBaseLearner(base_index=i, args=args)
        base_learners.append(virtual_learner)

    # meta learner initiation
    BHD3 = BalancingHyperparamDoublingDataDriven(m, dmin=1)
    selected_base_learners = []

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # Setting up the base learner for the episode
        if global_step%20==0:
            base_index = BHD3.sample_base_index()
            selected_base_learners.append(base_index)
            virtual_learner.set_base_learner(base_index, optimizer, args)
        if args.hparam_to_tune == "learning_rate":
            assert optimizer.param_groups[0]["lr"] == virtual_learner.get_hparam()
        elif args.hparam_to_tune == "epsilon":
            assert args.epsilon == virtual_learner.get_hparam()

        # ALGO LOGIC: put action logic here
        
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < args.epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    # Update baselearners distribution
                    episodic_return = info['episode']['r']
                    normalized_return = normalize_episodic_return(episodic_return, normalizer_const=1000)
                    BHD3.update_distribution(base_index, normalized_return)

                    writer.add_scalar("modelselection/metalearner_normalized_episodic_return", normalized_return, global_step)
                    writer.add_scalar(f"modelselection/baselearner_{base_index}_episodic_return", normalized_return, global_step)
                    writer.add_histogram("modelselection/num_plays", np.asarray(BHD3.num_plays), 0)
                        

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()