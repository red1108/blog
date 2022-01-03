def MCTS_viewer(node):
    print(node.game)
    print("N = {}, U = {}, V = {}".format(node.N, node.U, node.V))
    for c in node.child.values():
        MCTS_viewer(c)