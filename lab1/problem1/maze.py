"""
@brief Maze for Computer Lab1
@author Philipp Katterbach, Markus Pietschner
@date 5.11.2022

"""
import numpy as np
import matplotlib.pyplot as plt

class Maze:

    def __init__(self, scenario="a", stand_still=False, t=20):
        self.maze = self.__init_maze__()
        self.actions = self.__init_actions()
        self.states = self.__init_states__()
        self.rewards = self.__init_rewards__()
        self.PENALTY = -1000
        self.TRANSITION = -1
        self.GOAL = 1000
        self.entry = [0, 0]
        self.exit = [6, 7]
        self.scenario = scenario
        self.gamma = 0.95
        if scenario == "h":
            self.key = [0, 7]
        self.x_max = len(self.maze[0])
        self.y_max = len(self.maze)
        self.player_pos = self.entry
        self.minotaur_pos = self.exit
        self.stand_still = stand_still
        self.t = t
        self.sim = False

    # init methods
    def __init_maze__(self):
        return [[0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 1, 1, 1],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0]]
    def __init_actions(self):
        return ["left", "right", "up", "down", "stay"]

    def __init_states__(self):
        states = []
        for i in range(self.x_max):
            for j in range(self.y_max):
                for k in range(self.x_max):
                    for l in range(self.y_max):
                        if self.maze[i][j] != 1 and self.maze[k][l] != 1:
                            states.append((i, j, k, l))
        return states

    def __init_rewards__(self):
        rewards = [[]]
        for state in self.states:
            for action in self.actions:
                reward = 0
                x_player, y_player, x_min, y_min = state
                updated_pos = self.adjust_pos([x_player, y_player], action)
                x_player, y_player = updated_pos[0], updated_pos[1]
                min_actions = self.possible_actions([x_min, y_min])
                probabilities = self.get_trans_probabilities(state, min_actions)
                for i in range(len(min_actions)):
                    updated_pos = self.adjust_pos([x_min, y_min], min_actions[i])
                    x_min, y_min = updated_pos[0], updated_pos[1]
                    if [x_player, y_player] == [x_min, y_min]:
                        reward += probabilities[i] * self.PENALTY
                    elif [x_player, y_player] == self.exit:
                        reward += probabilities[i] * self.GOAL
                    elif not self.in_maze([x_player, y_player]):
                        reward += probabilities[i] * self.PENALTY
                    else:
                        reward += probabilities[i] * self.TRANSITION
        return rewards

    def in_maze(self, pos):
        x, y = pos[0], pos[1]
        if x >= 0 and x < self.x_max and y >= 0 and y < self.y_max:
            if self.maze[x][y] == 0:
                return True
        return False

    def get_good_actions(self, possible_actions, pos_player, pos_min):
        """Checks actions and gives out actions for Minotaur, which direct the Minotaur towards the player"""
        actions = []

        #TODO add possible actions

        return actions


    def get_trans_probabilities(self, state, actions):
        # adjust for better Minotaur in exercise h
        return [1/len(actions) for _ in actions]

    def get_value(self, pos):
        x, y = pos[0], pos[1]
        if x < 0 or y < 0 or x > self.x_max or y > self.y_max:
            return self.PENALTY
        if self.maze[x][y] == 1:
            return self.PENALTY
        if pos == self.exit:
            return self.GOAL
        else:
            return self.TRANSITION

    def possible_actions(self, pos=None):
        x = pos[0]
        y = pos[1]
        actions = []
        # check left
        if x > 0:
            if self.maze[x - 1][y] == 0:
                actions.append("left")
        # check right
        if x < self.x_max - 2:
            if self.maze[x + 1][y] == 0:
                actions.append("right")
        # check up
        if y > 0:
            if self.maze[x][y - 1] == 0:
                actions.append("up")
        # check down
        if y < self.y_max - 2:
            if self.maze[x][y + 1] == 0:
                actions.append("down")
        return actions

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

    def get_action(self, player, alg="dp"):
        """Returns the optimal action for either Minotaur or Player"""
        if player == "Player":
            if alg == "dp":
                val_ac = self.dp(self.player_pos, self.minotaur_pos, self.t)
                return val_ac[1]
        if player == "Minotaur":
            if self.scenario == "a":
                actions = self.possible_actions("Minotaur")
                i = np.random.randint(len(actions))
                return actions[i]

    def run(self):
        run = 0
        turn = "Minotaur"
        while run == 0:
            print("Step done")
            if not self.sim:
                min_action = self.get_action("Minotaur")
                player_action = self.get_action("Player")
                self.update_player_pos("Minotaur", min_action)
                self.update_player_pos("Player", player_action)
                run = self.check_result()
            else:
                action = self.get_action(turn)
                self.update_player_pos(turn, action)
                run = self.check_result()
                if turn == "Minotaur":
                    turn = "Player"
                else:
                    turn = "Minotaur"
        if run == 1:
            print("Minotaur caught the player")
        if run == 2:
            print("Player found the exit")

    def adjust_pos(self, pos, action):
        pos_new = pos.copy()
        if action == "left":
            pos_new[0] -= 1
        elif action == "right":
            pos_new[0] += 1
        elif action == "up":
            pos_new[1] -= 1
        elif action == "down":
            pos_new[1] += 1
        return pos_new

    #bs
    def dp(self, pos_player, pos_min, t):
        if pos_player == self.exit:
            return [1000, None]
        if pos_player == pos_min:
            return [-1000, None]
        if t == 0:
            return [0, None]
        actions_player = self.possible_actions(pos=pos_player)
        actions_min = self.possible_actions(pos=pos_min)
        prob = [1/3 for i in range(len(actions_min))]
        possible_values = []
        for i in range(len(actions_player)):
            value = 0
            pos_new_player = self.adjust_pos(pos_player, actions_player[i])
            for j in range(len(actions_min)):
                pos_new_min = self.adjust_pos(pos_min, actions_min[j])
                print(t)
                temp_val = self.dp(pos_new_player, pos_new_min, t - 1)
                temp_val[0] -= 1
                value += int(prob[j] * self.gamma * temp_val[0])
            possible_values.append(value)
        index = np.argmax(possible_values)
        return [possible_values[index], actions_player[index]]
