from abc import *


class Game(metaclass=ABCMeta):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        # return (next_state, reward, done, log)
        pass

    @abstractmethod
    def _allowed_actions(self):
        # return allowed actions with 1D list
        pass

    @abstractmethod
    def _check_game_end(self):
        # return true or false
        pass

    @abstractmethod
    def __str__(self):
        pass
