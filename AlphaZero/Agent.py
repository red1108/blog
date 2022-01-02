from abc import *
import random


class Agent(metaclass=ABCMeta):

    @abstractmethod
    def action(self, env):
        pass


class Agent_random:

    def __init__(self):
        pass

    def action(self, env):
        allowed_actions = env.allowed_actions()
        return random.choice(allowed_actions)


class Agent_DNN:
    def __init__(self, row=6, column=7):
        import numpy as np
        from os import environ
        environ["KERAS_BACKEND"] = "plaidml.keras.backend"
        import keras

        self.actor = keras.models.Sequential()
        self.actor.add(keras.layers.Dense(32, activation='relu'))
        self.actor.add(keras.layers.Dense(32, activation='relu'))
        self.actor.add(keras.layers.Dense(column))
        self.actor.compile('adam', 'mse')

        self.critic = keras.models.Sequential()
        self.critic.add(keras.layers.Dense(32, activation='relu'))
        self.critic.add(keras.layers.Dense(32, activation='relu'))
        self.critic.add(keras.layers.Dense(1, activation='tanh'))
        self.critic.compile('adam', 'mse')

    def action(self, env):
        allowed_actions = env.allowed_actions()
        return random.choice(allowed_actions)

