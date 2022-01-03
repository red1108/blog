from ConnectX import ConnectX
from Agent import Agent_random, Agent_DNN
from Play import play

if __name__ == '__main__':
    for _ in range(10):
        winner, step = play(ConnectX(), Agent_random(), Agent_DNN(), is_print=False)
        print("winner = {}, step = {}".format(winner, step))
