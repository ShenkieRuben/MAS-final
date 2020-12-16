from binary_tree import Tree


class MCTS():
    def __init__(self):
        print("Hello")


if __name__ == "__main__":
    d = 4
    total = 2**d
    tree = Tree()

    for i in range(total):
        tree.insert()

    tree.assign_vals_to_leafs(total)
    tree.debug_tree()
