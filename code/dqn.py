import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

#Creating the DQN environment - Atari benchmark
env = gym.make('BreakoutNoFrameskip-v4')
env = gym.wrappers.FrameStack(env, num_stack=4)

#Training the reinforcement learning agent
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=10000) # time stamps can be adjusted

#Save the model
model.save("../models/dqn_breakout")

#Pre-trained model can be loaded
# model = DQN.load("dqn_breakout")

#Evaluating the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

#Closing the environment
env.close()


# pip install gym[atari]