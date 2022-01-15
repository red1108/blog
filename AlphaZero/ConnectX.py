from Game import Game
import numpy as np
from copy import deepcopy


class ConnectX(Game):

    def __init__(self):
        self.col = 7
        self.row = 6
        self.state_size = self.col * self.row
        self.action_size = self.col
        self.state = np.zeros((self.row, self.col), dtype=np.int32)
        self.turn = 1
        self.duration = 0
        self.game_end = False
        self.winner = None
        self.__inarow = 4
        self.__dir = [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]
        self.__logs = []

    def reset(self):
        self.state = np.zeros((self.row, self.col), dtype=np.int32)
        self.turn = 1
        self.duration = 0
        self.game_end = False
        self.winner = None

    def step(self, action):
        if action not in self.allowed_actions():
            log = "Error({0}): not allowed action({1}) by {2}".format(self.duration, action, self.turn)
            self.__logs.append(log)
            return None, None, None, log

        if self.game_end:
            log = "Error({0}): game is already over".format(self.duration)
            self.__logs.append(log)
            return None, None, None, log

        row_pos = self.row - 1
        for i in range(self.row - 1, -1, -1):
            if self.state[i][action] == 0:
                row_pos = i
                break

        # increase game duration
        self.duration += 1

        log = "Message({0}): Player {1} mark at ({2}, {3})".format(self.duration, self.turn, row_pos, action)
        self.__logs.append(log)

        self.state[row_pos][action] = self.turn
        self.__logs.append(str(self))

        self.game_end |= self._check_game_end()

        if self.duration == self.col * self.row:
            self.game_end = True
            self.winner = 0

        reward = 1
        if self.game_end:
            if self.winner is None:
                self.winner = 1 if self.turn == 1 else -1

            reward = 10000
            newlog = "Message({0}): Player {1} win".format(self.duration, self.turn)
            self.__logs.append(newlog)
            log = ''.join([log, "\n" + newlog])

        self.turn = 3 - self.turn
        return self.state, reward, self.game_end, log

    def allowed_actions(self):
        allowed_actions = []
        for i in range(self.col):
            if self.state[0][i] == 0:
                allowed_actions.append(i)
        return allowed_actions

    def _check_game_end(self):
        if self.game_end:
            return True
        for i in range(self.row):
            for j in range(self.col):
                if self.__check_4(i, j, -1, 1) or self.__check_4(i, j, 0, 1) or self.__check_4(i, j, 1, 1) or self.__check_4(i, j, 1, 0):
                    return True
        return False

    def __check_4(self, x, y, dx, dy):
        if x + (self.__inarow - 1) * dx not in range(0, self.row): return False
        if y + (self.__inarow - 1) * dy not in range(0, self.col): return False

        for i in range(self.__inarow):
            if self.state[x][y] != self.turn:
                return False
            x = x + dx
            y = y + dy
        return True

    def __str__(self):
        ret = '=' * (self.col*2-1)
        ret += '\n'
        for i in range(self.row):
            for j in range(self.col):
                if self.state[i][j] == 0:
                    ret += '  '
                else:
                    ret += str(self.state[i][j])+' '
            ret += '\n'
        ret += '= ' * self.col
        return ret

    def __copy__(self):
        cls = self.__class__
        new_game = cls.__new__(cls)
        new_game.__dict__.update(self.__dict__)

        new_game.col = self.col
        new_game.row = self.row
        new_game.state_size = self.state_size
        new_game.action_size = self.action_size
        new_game.state = deepcopy(self.state)
        new_game.turn = self.turn
        new_game.duration = self.duration
        new_game.game_end = self.game_end
        new_game.winner = self.winner
        new_game.__inarow = self.__inarow
        new_game.__dir = deepcopy(self.__dir)
        new_game.__logs = deepcopy(self.__logs)
        return new_game

    def print_logs(self):
        for log in self.__logs:
            print(log)

    def get_state(self):
        ret = []
        for i in range(self.row):
            for j in range(self.col):
                ret.append(2-self.state[i][j])
                ret.append(self.state[i][j]-1)
        return np.reshape(ret, (1, 2 * self.col * self.row))

    def get_winner(self):
        return self.winner

    def get_player(self):
        return 3 - 2 * self.turn
