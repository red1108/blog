from Play import play
from copy import copy

def MCTS_all_viewer(node):
    print(node.game)
    print("N = {}, U = {}, V = {}, turn = {}".format(node.N, node.U, node.V, node.game.get_player()))
    for c in node.child.values():
        MCTS_all_viewer(c)

def MCTS_node_viewer(node, state_print=False):
    if state_print:
        print(node.game)
    print("N = {}, U = {}, V = {}, turn = {}".format(node.N, node.U, node.V, node.game.turn))
    for a, c in node.child.items():
        print("action = {}, N = {}, U = {}, V = {}, W={}, P={}".format(a, c.N, c.U, c.V, c.winner, c.game.get_player()))


def calculate_winrate(env, player1, player2, is_print=False, game_count=200):
    player1_win = 0
    player2_win = 0
    draw = 0
    step_sum=0
    for _ in range(game_count // 2):
        winner, step = play(copy(env), player1, player2, is_print)
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            draw += 1
        step_sum += step

    w1 = player1_win*2/game_count
    s1 = step_sum*2/game_count

    player1_win=0
    player2_win=0
    step_sum=0

    for _ in range(game_count // 2):
        winner, step = play(copy(env), player2, player1, is_print)
        if winner == 1:
            player2_win += 1
        elif winner == -1:
            player1_win += 1
        else:
            draw += 1
        step_sum += step
    w2 = player1_win*2/game_count
    s2 = step_sum*2/game_count

    return w1, s1, w2, s2, draw/game_count


def print_comparison(env, player1, player2, count, is_print=False):
    w1, s1, w2, s2, _ = calculate_winrate(env, player1, player2, game_count=count, is_print=is_print)
    print("{} vs {} => {}%, average = {}".format(player1.get_name(), player2.get_name(), w1 * 100, s1))
    print("{} vs {} => {}%, average = {}".format(player2.get_name(), player1.get_name(), w2 * 100, s2))
