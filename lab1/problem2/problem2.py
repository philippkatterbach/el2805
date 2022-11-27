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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import pdb
from math import cos,pi,sin
import random
from mpl_toolkits.mplot3d import Axes3D


# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 500       # Number of episodes to run for training
discount_factor = 1.    # Value of gamma
trace_factor = 0.7      #Value of lambda
learning_rate = 0.0001
eps = -1 #better
momentum = 0.7
slowed = False


# Reward
episode_reward_list = []  # Used to save episodes reward


def compute_Q(frequencies,weights,state):
    #frequencies: list of frequency vectors
    #weights: list containing 3 vectors for the 3 actions
    a0_weights = weights[0] #vector
    a1_weights = weights[1] #vector
    a2_weights = weights[2] #vector
    Q0 = 0
    Q1 = 0
    Q2 = 0
    for i in range(len(frequencies)):
        Q0 = Q0 + a0_weights[i]*cos(pi*np.dot(frequencies[i],state))
        Q1 = Q1 + a1_weights[i]*cos(pi*np.dot(frequencies[i],state))
        Q2 = Q2 + a2_weights[i]*cos(pi*np.dot(frequencies[i],state))
        

    return [Q0,Q1,Q2]

def compute_Q_grad(frequencies,state):
    #frequencies: list of frequency vectors
    Q_grad = [0]*len(frequencies)
    for i in range(len(frequencies)):
        Q_grad[i] = cos(pi*np.dot(frequencies[i],state))

    #This returns basically the value of the basis functions at state, idk if right

    return Q_grad




# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    # pdb.set_trace()
    x = (s[0] - low) / (high - low)
    return x

frequencies = [ [0,0],[0,1],[0,2],
                [1,0],[1,1],#[1,2],
                [2,0]#,[2,1],[2,2],
                      #[0,1],[0,2],
                #[1,0],[1,1],[1,2],
                #[2,0],[2,1],[2,1]
                ]
learning_rate_list = []
for i in range(len(frequencies)):
    if i == 0:
        learning_rate_list.append(1.0)
    else:
        learning_rate_list.append(1/np.linalg.norm(frequencies[i]))

learning_rate_matrix = learning_rate*np.diag(learning_rate_list)
#Assume correlation for lower frequecies, frequencies up to 2

weights = np.array([[0.0]*len(frequencies),[0.0]*len(frequencies),[0.0]*len(frequencies)])

# weights[0][1]=1
# Q = compute_Q(frequencies,weights,[1,0])
# Q_list = []
# for i in range(10):
#     Q = compute_Q(frequencies,weights,[0,i])
#     Q_list.append(Q[0])
# pdb.set_trace()

# Training process
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    state = scale_state_variables(env.reset())
    total_episode_reward = 0.
    z = np.array([[0.0]*len(frequencies),[0.0]*len(frequencies),[0.0]*len(frequencies)])
    v = np.array([[0.0]*len(frequencies),[0.0]*len(frequencies),[0.0]*len(frequencies)])
    print(i)
    j=0
    k = len(episode_reward_list)
    if k > 2 and episode_reward_list[k-1] > -190 and episode_reward_list[k-2] > -190 and episode_reward_list[k-3] > -190 and not slowed:
        slowed = True
        learning_rate = learning_rate/10

    while not done:

        # Take a random action
        # env.action_space.n tells you the number of actions
        # pdb.set_trace()
        # available
        j = j+1
        Q = compute_Q(frequencies,weights,state)

        if random.uniform(0, 1) > eps:
            action = Q.index(max(Q))
        else: 
            action = int(random.uniform(0, 3))
        # action = np.random.randint(0, k)
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, *_ = env.step(action)
        next_state = scale_state_variables(next_state)

        Q_next =compute_Q(frequencies,weights,next_state)

        if random.uniform(0, 1) > eps:
            action_next = Q_next.index(max(Q_next))
        else: 
            action_next = int(random.uniform(0, 3))

        Q_grad = compute_Q_grad(frequencies,state)
        #Compute next eligibility trace
        z[0] = discount_factor*trace_factor*z[0]
        z[1] = discount_factor*trace_factor*z[1]
        z[2] = discount_factor*trace_factor*z[2]
        if action == 0:
            z[0] = z[0] + Q_grad
        if action == 1:
            z[1] = z[1] + Q_grad
        if action == 2:
            z[2] = z[2] + Q_grad

        np.clip(z[0], -5, 5)
        np.clip(z[1], -5, 5)
        np.clip(z[2], -5, 5)

        temp_diff = reward + discount_factor*Q_next[action_next] - Q[action]
        v = momentum*v
        v[0] = v[0] + np.matmul(1/j*learning_rate_matrix,temp_diff*z[0])
        v[1] = v[1] + np.matmul(1/j*learning_rate_matrix,temp_diff*z[1])
        v[2] = v[2] + np.matmul(1/j*learning_rate_matrix,temp_diff*z[2])

        weights[0] = weights[0] + v[0]
        weights[1] = weights[1] + v[1]
        weights[2] = weights[2] + v[2]
        
       
       
        # Update episode reward
        total_episode_reward += reward
            
        # Update state for next iteration
        state = next_state

        if j == 200:
            done = True

        # if done:
        #     print(state)


    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    # print(j)
    # print(weights[0]-weights[1])

    # Close environment
env.close()
    

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

count = np.zeros(101)
Q_test_1 = np.zeros(101)
Q_test_2 = np.zeros(101)
Q_test_3 = np.zeros(101)
arg_max = np.zeros(101)
for k in range(0,101):
    count[k] = k/100
    Q_test = compute_Q(frequencies,weights,[count[k],0.0])
    Q_test_1[k] = Q_test[0]
    Q_test_2[k] = Q_test[1]
    Q_test_3[k] = Q_test[2]
    arg_max[k] = Q_test.index(max(Q_test))
plt.plot(count,Q_test_1)
plt.plot(count,Q_test_2)
plt.plot(count,Q_test_3)
plt.grid(alpha=0.3)
plt.show()

plt.plot(count,arg_max)
plt.grid(alpha=0.3)
plt.show()
# Q1 = np.zeros((101,101))
# Q2 = np.zeros((101,101))
# Q3 = np.zeros((101,101))

# X, Y = np.meshgrid(count, count)

# Q = compute_Q(frequencies,weights,[X,Y])
# Q1 = Q[0]
# Q2 = Q[1]
# Q3 = Q[1]


# fig = plt.figure(figsize=(4,4))

# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, Q1, rstride=1, cstride=1,cmap='jet', edgecolor = 'none')

pdb.set_trace()