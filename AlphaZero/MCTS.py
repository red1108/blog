from copy import copy
from math import sqrt
import random
import numpy as np

# U = V + C * prob * sqrt(N_tot)/(1+N)
C = 1.0


def agent_result(agent, game):
    probs, value = agent.result(game)
    allowed_action = game.allowed_actions()
    probs = [probs[i] for i in allowed_action]
    return game.allowed_actions(), probs, value


class Node:
    def __init__(self, game, mother=None, prob=0):
        self.game = game

        # dictionary of children
        # {action1:child1, action2:child2, ... }
        self.child = {}

        # Node's attractiveness index
        self.U = 0

        self.prob = prob
        self.nn_v = 0

        # visit count
        self.N = 0
        # expected V from MCTS
        self.V = 0

        # 1, -1 for winner, 0 for tie, else None.
        self.winner = self.game.get_winner()

        if self.winner is not None:
            self.V = self.winner * self.game.get_player()
            self.U = 0 if self.winner == 0 else self.V * float('inf')

        self.mother = mother

    def create_child(self, actions, probs):
        # create a dictionary of children

        games = [copy(self.game) for _ in actions]

        for action, game in zip(actions, games):
            game.step(action)

        self.child = {a:Node(g, self, p) for a, g, p in zip(actions, games, probs)}

    def explore(self, agent):

        if self.game.winner is not None:
            raise ValueError("ERROR: the game is already ended with winner {0}".format(self.game.get_winner()))

        current = self

        # explore to the leaf
        while current.child and current.winner is None:

            child = current.child
            max_U = max(c.U for c in child.values())

            # Choose the best action.
            actions = [a for a, c in child.items() if c.U == max_U]
            if len(actions) == 0:
                raise ValueError("ERROR: have no children")

            action = random.choice(actions)

            if max_U == -float('inf') or max_U == float('inf'):
                # multiplied by -1 because it's from next player
                current.U = -max_U
                current.V = -1.0 if max_U == float('inf') else 1.0
                break

            current = child[action]

        # if node hasn't been expanded
        if not current.child and current.winner is None:
            # agent outputs results from next player's point of view.
            # thus multiplied by -1
            allowed_actions, probs, value = agent_result(agent, current.game)
            current.nn_v = -value
            current.create_child(allowed_actions, probs)
            current.V = -float(value)

        current.N += 1

        # Update U from backpropagation
        while current.mother:
            mother = current.mother
            mother.N += 1
            # moving average
            mother.V += (-current.V - mother.V) / mother.N

            # Update sibling's U
            for sibling in mother.child.values():
                if abs(sibling.U) is not float("inf"):
                    sibling.U = sibling.V + C * float(sibling.prob) * sqrt(mother.N) / (1 + sibling.N)

            current = current.mother

    def next(self, temperature=1.0):
        if self.game.get_winner() is not None:
            raise ValueError("ERROR: the game is already ended with winner {0}".format(self.game.get_winner()))

        if not self.child:
            print("Debug: ")
            print(self.game)
            raise ValueError("have no children and game hasn't ended")

        child = self.child

        max_U = max(c.U for c in child.values())

        if max_U == float('inf'):
            prob = [1.0 if c.U == max_U else 0 for c in child.values()]
        else:
            # Selection based on the number of searches
            max_N = max(c.N for c in child.values()) + 1
            prob = [(c.N / max_N) ** (1 / temperature) for c in child.values()]

        # normalize
        prob = np.array(prob)
        if sum(prob) > 0:
            prob = prob / sum(prob)
        else:
            prob = np.array([1.0 / len(prob) for _ in prob])

        nn_prob = [c.prob for c in child.values()]
        next_state = random.choices(list(child.values()), weights=prob)[0]

        return next_state, self.game.allowed_actions(), (-self.V, -self.nn_v, prob, nn_prob)

    def detach_mother(self):
        del self.mother
        self.mother = None