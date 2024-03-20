import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

#Creating the PPO environment - MuJoCo benchmark
env = gym.make('HalfCheetah-v3')  # Replace with any Mujoco environment

#Training the reinforcement learning agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000) 

#Save the model
model.save("../models/ppo_mujoco")

#Pre-trained model can be loaded
# model = PPO.load("ppo_mujoco")

#Evaluating the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

#Closing the environment
env.close()



# pip install gym[mujoco]