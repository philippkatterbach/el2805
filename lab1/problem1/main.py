from maze import dynamic_programming
from maze import draw_maze
import numpy as np


T = 20
maze = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0]]
start = (0, 0)
maze = np.array(maze)

V, policy = dynamic_programming(maze, 20)
path = maze.simulate(start, policy, 'DynProg')
draw_maze(path)
