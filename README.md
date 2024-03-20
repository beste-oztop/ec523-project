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

- `code/`: Contains the implementation code for DQN and PPO algorithms.
- `models/`: Directory to store trained models.
- `README.md`: Overview of the project, instructions for running the code, and any additional information for users.

## Running the Code 

To run the implemented algorithms:

1. Clone the project repository to your local machine.
2. Navigate to the `code/` directory.
3. Run the appropriate Python script for the desired algorithm (`dqn.py` for DQN, `ppo.py` for PPO).
4. Follow the instructions provided in the script to configure hyperparameters and choose the benchmark environment (Atari or Mujoco).
5. After training, evaluate the trained models using the provided evaluation script or directly within the training script.

## Contributors ##

- Aida Afshar Mohammadian
- Beste Oztop
