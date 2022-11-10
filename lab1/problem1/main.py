from maze import dynamic_programming
from maze import draw_maze
from maze import Maze
from maze import animate_solution
import numpy as np


T = 20
maze = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0]]
start = (0,0)
maze = np.array(maze)
env = Maze(maze)

V, policy = dynamic_programming(env, 20)
path = env.simulate(start, policy, 'DynProg')
animate_solution(maze, path)