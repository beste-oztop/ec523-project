# A Comparative Study of Deep Q-Networks and Proximal Policy Optimization in Reinforcement Learning 

## Project Description

The goal of this project is to investigate the Neural Network architectures utilized in two of the standard Deep Reinforcement Learning (DRL) algorithms. We will implement each of them and test their performances with the existing benchmark applications. We plan to go for one off-policy and one on-policy DRL algorithm and to provide theoretical and practical comparisons between them throughout the project.

## Implemented Algorithms 

1. Deep Q-Networks (DQN) 
  - Implementation with Convolutional Neural Network (CNN) to support training with image input.
  -  Design choices regarding the implementation of the experience replay buffer will be explored.

2. Proximal Policy Optimization (PPO) 
  - Implementation with a fully connected architecture in the network.
  - PPO requires careful consideration and implementation tricks, as per the pseudocode available in the original paper.


## Benchmark Applications 
The benchmark applications selected for this project are:

- Atari Games
  - The Atari environment provides a suite of classic Atari 2600 games as benchmark environments for reinforcement learning.
  - These games provide diverse challenges and require agents to learn complex strategies from pixel inputs.
  - The success metrics for the Atari benchmarks will include average episode reward, training stability, and generalization to unseen game states.
- Mujoco Environments
  -  The Mujoco physics simulation engine offers a variety of continuous control tasks, including locomotion and manipulation tasks.
  -  These environments require agents to learn precise motor control and decision-making in continuous action spaces.
  - Success metrics for Mujoco environments will include average episode reward, sample efficiency, and robustness to perturbations.


## Repository Structure 

The project repository will have the following structure:

- `algorithms/`: Contains the implementation code for DQN and PPO algorithms.
- `buffers/`: Contains the replay buffer implementation
- `code/`: Contains the ready-to-use implementation code for DQN and PPO algorithms.
- `networks/`: Directory to store trained networks.
- `README.md`: Overview of the project, instructions for running the code, and any additional information for users.

## Running the Code 

To run the implemented algorithms:

1. Clone the project repository to your local machine.
2. Make sure you created a virtual environment with the necessary `python` libraries; namely, `gymnasium`, `Box2D` (with potential system-level dependencies), `torch`, `tyro`, `tensorboard` and `wandb`. One can run `install_libs.sh` in the virtual environment to install these libraries easily.
3. Run the `main.py` with the command line arguments of your choice. Explanations and the default versions are as follows:
  --alg_name: Name of the RL algorithm to use (default is "PPO").
  --exp_name: Name of the experiment (default is the name of the Python file).
  --seed: Seed of the experiment (default is 1).
  --torch_deterministic: Toggle for deterministic PyTorch behavior (default is True).
  --cuda: Toggle for enabling CUDA (default is True).
  --track: Toggle for tracking the experiment with Weights and Biases (default is False).
  --wandb_project_name: Name of the Weights and Biases project (default is "ec523").
  --wandb_entity: Entity (team) of the Weights and Biases project (default is None).
  --capture_video: Toggle for capturing videos of agent performances (default is False).
  --save_model: Toggle for saving the model into the runs/{run_name} folder (default is False).
  --upload_model: Toggle for uploading the saved model to Hugging Face Hub (default is False).
  --hf_entity: User or org name of the model repository from the Hugging Face Hub (default is empty string).
  --env_id: ID of the environment (default is "CartPole-v1").
  --total_timesteps: Total timesteps of the experiment (default is 500000).
  --learning_rate: The learning rate of the optimizer (default is 2.5e-4).
  --num_envs: Number of parallel game environments (default is 1).
  --buffer_size: Replay memory buffer size (default is 10000).
  --gamma: Discount factor gamma (default is 0.99).
  --tau: Target network update rate (default is 1.0).
  --target_network_frequency: Timesteps it takes to update the target network (default is 500).
  --batch_size: The batch size of the sample from the replay memory (default is 128).
  --start_e: Starting epsilon for exploration (default is 1).
  --end_e: Ending epsilon for exploration (default is 0.05).
  --exploration_fraction: Fraction of total timesteps it takes from start-e to go end-e (default is 0.5).
  --learning_starts: Timestep to start learning (default is 10000).
  --train_frequency: Frequency of training (default is 10).

## Contributors ##

- Aida Afshar Mohammadian
- Beste Oztop
