def play(env, player1, player2, is_print=False):
    step = 0
    while True:
        agent = player1 if step%2 == 0 else player2
        _state, _reward, done, _log = env.step(agent.action(env))
        step += 1
        if done:
            break
    if is_print:
        env.print_logs()
    return env.get_winner(), step

