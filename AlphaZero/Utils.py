from Play import play
from copy import copy

def MCTS_viewer(node):
    print(node.game)
    print("N = {}, U = {}, V = {}".format(node.N, node.U, node.V))
    for c in node.child.values():
        MCTS_viewer(c)


def calculate_winrate(env, player1, player2, is_print=False, game_count=100):
    player1_win = 0
    player2_win = 0
    draw = 0
    for _ in range(game_count // 2):
        winner, step = play(copy(env), player1, player2, is_print)
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            draw += 1

    for _ in range(game_count // 2):
        winner, step = play(copy(env), player2, player1, is_print)
        if winner == 1:
            player2_win += 1
        elif winner == -1:
            player1_win += 1
        else:
            draw += 1

    return player1_win/game_count, player2_win/game_count, draw/game_count
