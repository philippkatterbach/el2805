"""
@brief Maze for Computer Lab1
@author Philipp Katterbach, Markus Pietschner
@date 5.11.2022

"""
import numpy as np
import matplotlib.pyplot as plt

OPEN = 0
WALL = 1

class Maze:

    def __init__(self, scenario="a"):
        self.maze = [[0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 1, 1],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0]]
        self.entry = [0, 0]
        self.exit = [6, 7]
        if scenario == "h":
            self.key = [0, 7]
        self.x_max = len(self.maze[0])
        self.y_max = len(self.maze)
        self.player_pos = self.entry
        self.minotaur_pos = self.exit

    def update_player_pos(self, player, action):
        if player == "Minotaur":
            pos = self.minotaur_pos
        else:
            pos = self.player_pos
        # no need to check if an action is valid since we checked earlier
        if action == "left":
            pos[1] -= 1
        elif action == "right":
            pos[1] += 1
        elif action == "up":
            pos[0] -= 1
        elif action == "down":
            pos[0] += 1

    def possible_actions(self, player=None, pos=None):
        if player == "Minotaur":
            x = self.minotaur_pos[0]
            y = self.minotaur_pos[1]
        elif player == "Player":
            x = self.player_pos[0]
            y = self.player_pos[1]
        elif pos is not None:
            x = pos[0]
            y = pos[1]
        actions = []
        # check left
        if x > 0:
            if self.maze[x - 1][y] == 0:
                actions.append("left")
        # check right
        if x < self.x_max:
            if self.maze[x + 1][y] == 0:
                actions.append("right")
        # check up
        if y > 0:
            if self.maze[x][y - 1] == 0:
                actions.append("up")
        # check down
        if y < self.y_max:
            if self.maze[x][y + 1] == 0:
                actions.append("down")
        return actions

    def minotaur_move(self, scenario="a"):
        if scenario == "a":
            actions = self.possible_actions("Minotaur")
            i = np.random.randint(len(actions))
            self.update_player_pos("Minotaur", actions[i])

    def check_result(self):
        """Evaluates the current state of the players in the maze.
        0: No result yet, game goes on
        1: Player failed
        2: Player won
        """
        if self.player_pos == self.minotaur_pos:
            return 1
        elif self.player_pos == self.exit:
            return 2
        return 0





