from abc import *
import random
import numpy as np

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


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
        self.actor.add(layers.Dense(column, activation='softmax'))
        self.actor.compile('adam', 'mse')

        self.critic = tf.keras.models.Sequential()
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(32, activation='relu'))
        self.critic.add(layers.Dense(1, activation='tanh'))
        self.critic.compile('adam', 'mse')

    def action(self, env):
        state = env.get_state()
        nn_probs = self.actor(state).numpy()
        nn_probs = np.reshape(nn_probs, (self.column, ))
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

