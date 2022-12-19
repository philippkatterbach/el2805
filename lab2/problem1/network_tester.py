import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from collections import deque
import torch.nn as nn
import torch.optim as optim
import random
import pdb
class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size,32)
        self.input_layer_activation = nn.ReLU()

        self.layer1 = nn.Linear(32,32)
        self.layer1_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(32,output_size)
    def forward(self, x):
        l1 = self.input_layer(x)
        l1_act = self.input_layer_activation(l1)
        l2 = self.layer1(l1_act)
        l2_act = self.layer1_activation(l2)
        output = self.output_layer(l2_act)
        return output
network = torch.load('neural-network-1.pth')
print('Network model: {}'.format(network))


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y
class ExReBu(object):
    def __init__(self,max_length = 1000):
        self.buffer = deque(maxlen=max_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self,num_elem = 50):
        if num_elem > len(self.buffer):
            print("Error: Buffer is too small")

        indices = np.random.choice(len(self.buffer),num_elem,replace=False)

        batch = [self.buffer[i] for i in indices]

        return zip(*batch)

env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 500                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
buffer_len = 20000
training_batch_size = 100
update_time = int(buffer_len/training_batch_size)
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:

        #Take e-greedy action of network
        state_tensor = torch.tensor([state],requires_grad=False, dtype=torch.float32)
        action_values = network(state_tensor)
        val,index = action_values.max(1)
        action = index.item()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        exp = (state, action, reward,next_state,done)


        # Update episode reward
        total_episode_reward += reward

        #Update net


        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()