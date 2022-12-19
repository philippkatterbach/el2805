# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
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

# Import and initialize the discrete Lunar Laner Environment
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
learning_rate = 1e-4
eps_max = 0.99
eps_min = 0.05

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)

#Network

buffer = ExReBu(max_length=buffer_len)

network = MyNetwork(input_size=dim_state,output_size=n_actions)

optimizer = optim.Adam(network.parameters(),lr=learning_rate)

target_network = MyNetwork(input_size=dim_state,output_size=n_actions)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
update_count = 0
for i in EPISODES:
    eps = max(eps_min,eps_max - (eps_max-eps_min)*i/N_episodes)
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:

        #Take e-greedy action of network
        if random.uniform(0, 1) < eps:
            action = np.random.randint(0, n_actions + 4)
            #do nothing with 50%
            if action > 3:
                action = 0
        else: 
            state_tensor = torch.tensor([state],requires_grad=False, dtype=torch.float32)
            action_values = network(state_tensor)
            val,index = action_values.max(1)
            action = index.item()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        exp = (state, action, reward,next_state,done)

        buffer.append(exp)

        # Update episode reward
        total_episode_reward += reward

        #Update net

        if len(buffer) >= training_batch_size:
            optimizer.zero_grad()
            states,actions,rewards,next_states,dones = buffer.sample_batch(training_batch_size)
            states_values_all = network(torch.tensor(states,requires_grad=False,dtype=torch.float32))
            # pdb.set_trace()
            # action_values = torch.zeros(1,training_batch_size,requires_grad=False,dtype=torch.float32)
            # target_values = torch.zeros(1,training_batch_size,requires_grad=False,dtype=torch.float32)
            indizes = np.arange(training_batch_size,dtype=np.int32)
            # pdb.set_trace()

            ###############################################
            #Was hier folgt ist mies gecoded, pls fix TODO
            '''Wann muss requires grad und wann nicht? Kann ich die state und target values so erstellen?'''
            states_values = torch.zeros(training_batch_size,1,dtype=torch.float32,requires_grad=False)
            target_values = torch.zeros_like(torch.tensor(states_values),requires_grad=False)
            for i in range(training_batch_size):
                states_values[i] = states_values_all[i,actions[i]]
                if dones[i]:
                    target_values[i] = rewards[i]
                else:
                    next_QValues = target_network(torch.tensor([next_states[i]],requires_grad=False,dtype=torch.float32))
                    # pdb.set_trace()
                    Q_next, action_next = next_QValues.max(1)
                    target_values[i] = rewards[i] + discount_factor * Q_next[0].item()
            ###############################################


            loss = nn.functional.mse_loss(states_values, target_values)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(),1.0)
            optimizer.step()

        # Update state for next iteration
        state = next_state
        t+= 1
        update_count+= 1

        #updating target network
        if update_count >= update_time:
            target_network = network
            update_count = 0

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

# torch.save(target_network, 'neural-network-1.pth')

pdb.set_trace()