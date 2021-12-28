def agent_random(env):
    import random
    valid_actions = env._allowed_actions()
    return random.choice(valid_actions)


class