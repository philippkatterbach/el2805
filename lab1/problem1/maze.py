"""
@brief Maze for Computer Lab1
@author Philipp Katterbach, Markus Pietschner
@date 5.11.2022

"""
import numpy as np
import matplotlib.pyplot as plt

class Maze:

    def __init__(self, scenario="a", stand_still=False):
        # statics
        self.PENALTY = -1000
        self.TRANSITION = -1
        self.GOAL = 1000
        self.entry = [0, 0]

        self.maze = self.__init_maze__()
        self.x_max = len(self.maze[0])
        self.y_max = len(self.maze)
        self.exit = [self.x_max - 1, self.y_max - 1]
        self.actions = self.__init_actions()
        self.states = self.__init_states__()
        self.rewards = self.__init_rewards__()

        self.scenario = scenario
        self.gamma = 0.95
        if scenario == "h":
            self.key = [0, 7]

        self.player_pos = self.entry
        self.minotaur_pos = self.exit
        self.stand_still = stand_still
        self.T = 20
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
                        if self.maze[j][i] != 1 and self.maze[l][k] != 1:
                            states.append((i, j, k, l))
        return states

    def __init_rewards__(self):
        rewards = [[None for y in range(len(self.actions))] for x in range(len(self.states))]
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                reward = 0
                x_player, y_player, x_min, y_min = self.states[i]
                updated_pos = self.adjust_pos([x_player, y_player], self.actions[j])
                if self.in_maze(updated_pos):
                    x_player, y_player = updated_pos[0], updated_pos[1]
                    min_actions = self.possible_actions([x_min, y_min], player="Minotaur")
                    probabilities = self.get_trans_probabilities(self.states[i], min_actions)
                    for k in range(len(min_actions)):
                        updated_pos = self.adjust_pos([x_min, y_min], min_actions[k])
                        x_min, y_min = updated_pos[0], updated_pos[1]
                        if [x_player, y_player] == [x_min, y_min]:
                            reward += probabilities[k] * self.PENALTY
                        elif [x_player, y_player] == self.exit:
                            reward += probabilities[k] * self.GOAL
                        else:
                            reward += probabilities[k] * self.TRANSITION
                    rewards[i][j] = reward
                else:
                    rewards[i][j] = -1000
        return rewards

    def dp(self):
        V = [[None for y in range(self.T + 1)] for x in range(len(self.states))]
        policy = V.copy()
        Q = self.rewards.copy()
        for i in range(len(self.states)):
            arg_max = 0
            val_max = -1000
            for j in range(len(self.actions)):
                if Q[i][j] > val_max:
                    val_max = Q[i][j]
                    arg_max = j
            V[i][self.T] = val_max
            policy[i][self.T] = arg_max

        for t in range(self.T - 1, -1, -1):
            for i in range(len(self.states)):
                x_player, y_player, x_min, y_min = self.states[i]
                arg_max = 0
                val_max = -1000
                for j in range(len(self.actions)):
                    new_pos = self.adjust_pos([x_player, y_player], self.actions[j])
                    if self.in_maze(new_pos):
                        s_new = self.states.index((new_pos[0], new_pos[1], x_min, y_min))
                        Q[i][j] = self.rewards[i][j] + V[s_new][t + 1]
                    else:
                        Q[i][j] = self.rewards[i][j] - 1000
                    if Q[i][j] > val_max:
                        val_max = Q[i][j]
                        arg_max = j
                V[i][t] = val_max
                policy[i][t] = arg_max
        return V, policy

    def in_maze(self, pos):
        x, y = pos[0], pos[1]
        if x >= 0 and x < self.x_max and y >= 0 and y < self.y_max:
            if self.maze[y][x] == 0:
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

    def possible_actions(self, pos=None, player="Player"):
        x = pos[0]
        y = pos[1]
        actions = []
        # check left
        if x > 0:
            if self.maze[y][x-1] == 0:
                actions.append("left")
        # check right
        if x < self.x_max - 1:
            if x >= 6:
                a = 0
            if self.maze[y][x+1] == 0:
                actions.append("right")
        # check up
        if y > 0:
            if self.maze[y-1][x] == 0:
                actions.append("up")
        # check down
        if y < self.y_max - 1:
            if self.maze[y+1][x] == 0:
                actions.append("down")
        if player == "Player":
            actions.append("stay")
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
        if player == "Minotaur":
            if self.scenario == "a":
                actions = self.possible_actions(pos=self.minotaur_pos, player="Minotaur")
                i = np.random.randint(len(actions))
                return actions[i]
        if player == "Player":
            _, policy = self.dp()
            s = (self.player_pos[0], self.player_pos[1], self.minotaur_pos[0], self.minotaur_pos[1])
            state = self.states.index(s)
            player_action = policy[state][0]
            return player_action

    def update_player_pos(self, player, action):
        if player == "Minotaur":
            self.minotaur_pos = self.adjust_pos(self.minotaur_pos, action)
        else:
            self.player_pos = self.adjust_pos(self.player_pos, action)

    def run(self):
        run = 0
        turn = "Minotaur"
        while run == 0:
            print("Step done")
            if not self.sim:
                player_action = self.get_action("Player")
                min_action = self.get_action("Minotaur")
                self.update_player_pos("Minotaur", min_action)
                self.update_player_pos("Player", player_action)
                print("Minotaur position: " + str(self.minotaur_pos))
                print("Player position: " + str(self.player_pos))
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
