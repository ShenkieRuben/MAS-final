# A very simple binary tree class for the MCTS exercise.
# Uses a dictionary as data structure.

import numpy as np
from numpy import random
from numpy.core.fromnumeric import size


class Node:
    def __init__(self) -> None:

        self.t = 0
        self.n = 0


class Tree:
    def __init__(self) -> None:
        self.data = {}
        self.leaves = set()
        self.root = None

    def get_root(self):
        if self.data:
            return self.data[0]
        else:
            return None

    def get_parent(self, node_key):
        # Right child has even node key
        if node_key % 2 == 0:
            parent = (node_key - 2) / 2
        else:
            parent = (node_key - 1) / 2

        return parent

    def get_left_child(self, node_key):
        return node_key * 2 + 1

    def get_right_child(self, node_key):
        return node_key * 2 + 2

    def insert(self):
        if not self.get_root():
            self.data[0] = Node()
            self.leaves.add(0)
        else:
            # Take an arbitrary node (leaf) and give it children
            node_key = self.leaves.pop()
            left = node_key * 2 + 1
            right = node_key * 2 + 2
            self.data[left] = Node()
            self.data[right] = Node()

            self.leaves.add(left)
            self.leaves.add(right)

    def debug_tree(self):
        for key, node in self.data.items():
            node_t = node.t
            node_n = node.n

            # if node_n == -1 or key == 0:
            if node_n != 0:
                print(f"Node {key} with value {node_t/node_n}")

    def print_max(self):
        max = -1
        node_key = -1
        for key, node in self.data.items():
            if key in self.leaves:
                xbar = node.t / node.n
                if xbar > max:
                    max = xbar
                    node_key = key

        print(f"Max node is {node_key} with value {max}")

    def pp_tree(self, node_key, level):
        if node_key >= 0 and node_key in self.data:
            left = self.get_left_child(node_key)
            right = self.get_right_child(node_key)
            self.pp_tree(right, level + 1)
            print(' ' * 6 * level + '->',
                  f"{node_key}, {self.data[node_key].t}, {self.data[node_key].n}")
            self.pp_tree(left, level+1)

    def assign_vals_to_leafs(self, total_leafs):
        # TODO: change this to real numbers.
        # vals = np.random.uniform(0, 100, total_leafs)
        vals = np.random.randint(0, 100, total_leafs)
        temp_leaves = self.leaves.copy()
        for random_val in vals:
            leaf = temp_leaves.pop()
            self.data[leaf].t = -random_val
            self.data[leaf].n = -1
