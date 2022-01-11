from Play import play
from copy import copy

def MCTS_all_viewer(node):
    print(node.game)
    print("N = {}, U = {}, V = {}, turn = {}".format(node.N, node.U, node.V, node.game.get_player()))
    for c in node.child.values():
        MCTS_all_viewer(c)

def MCTS_node_viewer(node):
    #print(node.game)
    print("N = {}, U = {}, V = {}".format(node.N, node.U, node.V))
    for a, c in node.child.items():
        print("action = {}, N = {}, U = {}, V = {}".format(a, c.N, c.U, c.V))


def calculate_winrate(env, player1, player2, is_print=False, game_count=200):
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
    w1 = player1_win*2/game_count

    player1_win=0
    player2_win=0

    for _ in range(game_count // 2):
        winner, step = play(copy(env), player2, player1, is_print)
        if winner == 1:
            player2_win += 1
        elif winner == -1:
            player1_win += 1
        else:
            draw += 1
    w2 = player1_win*2/game_count

    return w1, w2, draw/game_count
