from ConnectX import ConnectX

if __name__ == '__main__':
    env = ConnectX()
    env.step(3)
    env.step(4)
    print(env)