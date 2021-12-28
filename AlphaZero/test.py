from ConnectX import ConnectX
from Agent import agent_random

if __name__ == '__main__':
    env = ConnectX()
    while True:
        _state, _reward, done, _log = env.step(agent_random(env))
        if done:
            break
    env.print_logs()