from maze import dynamic_programming
from maze import draw_maze
from maze import Maze
from maze import animate_solution
from maze import value_iteration
import numpy as np


T = 20
maze = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 2, 0, 0]]
start = (0,0,6,5)
maze = np.array(maze)


# define values for value iteration
gamma = 0.99
epsilon = 0.1

# probability for geometric distribution with mean = 30
p = 1/30

# create environment
env = Maze(maze, poison=True)
V, policy = value_iteration(env, gamma, epsilon)

path = env.simulate(start, policy, 'ValIter')
animate_solution(maze, path)