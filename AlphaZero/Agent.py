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
        valid_actions = env.allowed_actions()
        return random.choice(valid_actions)
