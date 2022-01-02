from ConnectX import ConnectX
from Agent import Agent_random, Agent_DNN
from Play import play

if __name__ == '__main__':
    winner, step = play(ConnectX(), Agent_random(), Agent_DNN(), is_print=True)
    print("winner = {}, step = {}".format(winner, step))