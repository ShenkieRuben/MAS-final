# A very simple binary tree class for the MCTS exercise.
# Uses a dictionary as data structure.

import numpy as np
from numpy import random
from numpy.core.fromnumeric import size


class Node:
    def __init__(self) -> None:

        self.t = 0
        self.n = 0
        self.val = 0

        # self.left = None
        # self.right = None


class Tree:
    def __init__(self) -> None:
        self.tree = {}
        self.leaves = set()
        self.root = None

    def get_root(self):
        if self.tree:
            return self.tree[0]
        else:
            return None

    def get_left_child(self, node_key):
        return self.tree[node_key * 2 + 1]

    def get_right_child(self, node_key):
        return self.tree[node_key * 2 + 2]

    def insert(self):
        if not self.get_root():
            self.tree[0] = Node()
            self.leaves.add(0)
        else:
            # Take an arbitrary node (leaf) and give it children
            node_key = self.leaves.pop()
            left = node_key * 2 + 1
            right = node_key * 2 + 2
            self.tree[left] = Node()
            self.tree[right] = Node()

            self.leaves.add(left)
            self.leaves.add(right)

    def debug_tree(self):
        for key, node in self.tree.items():
            node_val = node.val
            node_t = node.t
            node_n = node.n

            print(f"Node {key} with value {node_val}")
            print(f"The value of t is {node_t}")
            print(f"The value of n is {node_n}")

    def assign_vals_to_leafs(self, total_leafs):
        vals = np.random.uniform(0, 100, total_leafs)
        temp_leaves = self.leaves.copy()
        for random_val in vals:
            leaf = temp_leaves.pop()
            self.tree[leaf].val = random_val
