from abc import *
import random
import numpy as np

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from copy import copy


class Agent(metaclass=ABCMeta):

    @abstractmethod
    def action(self, env):
        pass


class Agent_random(Agent):

    def __init__(self):
        pass

    def action(self, env):
        allowed_actions = env.allowed_actions()
        return random.choice(allowed_actions)


class Agent_expert(Agent):
    def __init__(self):
        pass

    def state_value(self, board, player):
        row = len(board)
        col = len(board[0])
        value = 0
        for i in range(row):
            for j in range(col):
                value += self.reward_dir(board, player, i, j, -1, 1)
                value += self.reward_dir(board, player, i, j, 0, 1)
                value += self.reward_dir(board, player, i, j, 1, 1)
                value += self.reward_dir(board, player, i, j, 1, 0)
                value -= self.reward_dir(board, 3 - player, i, j, -1, 1)
                value -= self.reward_dir(board, 3 - player, i, j, 0, 1)
                value -= self.reward_dir(board, 3 - player, i, j, 1, 1)
                value -= self.reward_dir(board, 3 - player, i, j, 1, 0)
        return value

    def reward_dir(self, board, player, x, y, dx, dy):
        reward = [0, 1, 5, 100, 100000]
        row = len(board)
        col = len(board[0])
        if x + 3 * dx < 0 or x + 3 * dx >= row:
            return 0
        if y + 3 * dy < 0 or y + 3 * dy >= col:
            return 0
        count = 0
        for i in range(4):
            if board[x][y] == player:
                count = count + 1
            elif board[x][y] == 3 - player:
                return 0
            x = x + dx
            y = y + dy
        return reward[count]

    def action(self, env):
        valid_moves = env.allowed_actions()
        values = []
        for a in valid_moves:
            temp = copy(env)
            temp.step(a)
            values.append(self.state_value(temp.state, env.turn))

        actions = [valid_moves[i] for i in range(len(valid_moves)) if values[i] == max(values)]
        return random.choice(actions)


class Agent_NN(Agent, metaclass=ABCMeta):

    @abstractmethod
    def result(self, env):
        pass


class Agent_DNN(Agent_NN):
    def __init__(self, row=6, column=7):
        self.row = row
        self.column = column

        self.actor = tf.keras.models.Sequential()
        self.actor.add(layers.Dense(32, activation='relu', input_shape=(42,)))
        self.actor.add(layers.Dense(32, activation='relu'))
        self.actor.add(layers.Dense(32, activation='relu'))
        self.actor.add(layers.Dense(32, activation='relu'))
        self.actor.add(layers.Dense(column, activation='softmax'))
        self.actor.compile('adam', 'mse')

        self.critic = tf.keras.models.Sequential()
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(1, activation='tanh'))
        self.critic.compile('adam', 'mse')

    def action(self, env):
        state = env.get_state()
        nn_probs = self.actor(state).numpy()
        nn_probs = np.reshape(nn_probs, (self.column,))
        allowed_actions = env.allowed_actions()
        max_p = max(nn_probs[i] for i in allowed_actions)
        actions = [i for i in allowed_actions if nn_probs[i] == max_p]

        return random.choice(actions)

    def result(self, env):
        state = env.get_state()
        nn_probs = self.actor(state).numpy()
        nn_probs = np.reshape(nn_probs, (self.column,))

        nn_value = self.critic(state).numpy()
        return nn_probs, nn_value
