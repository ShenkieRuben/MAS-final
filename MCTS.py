from random import random
from binary_tree import Tree
import math
import numpy as np


class MCTS():
    def __init__(self, c, d, max_iters, max_rollouts, nr_iters):
        self.c = c
        self.d = d
        self.max_iters = max_iters
        self.max_rollouts = max_rollouts
        self.nr_iters = nr_iters

        num_leaves = 2 ** self.d

        self.tree = Tree()
        for _ in range(num_leaves):
            self.tree.insert()

        self.tree.assign_vals_to_leafs(num_leaves)

    def compute_UCB(self, node_key):

        if node_key not in self.tree.data:
            return math.inf

        t = self.tree.data[node_key].t
        n = self.tree.data[node_key].n

        parent = self.tree.get_parent(node_key)
        visit_parent = self.tree.data[parent].n

        if n > 0:
            return t/n + self.c * math.sqrt((math.log(visit_parent)/n))
        else:
            return math.inf

    # Perform the tree policy and construct a path from the root to the most promising leaf node (with highest finite UCB).
    # Return the snowcap leaf with the highest UCB.
    def select_node(self):
        # Always start at the root
        node_key = 0
        temp_node = node_key
        path = []

        # The root
        if self.tree.data[node_key].n == 0:
            path.append(node_key)
            return node_key, path

        # Larger than 0 to prevent computing UCB values on leaves.

        while(node_key in self.tree.data and self.tree.data[node_key].n > 0):
            temp_node = node_key
            path.append(node_key)
            left = self.tree.get_left_child(node_key)
            right = self.tree.get_right_child(node_key)

            ucb_left = self.compute_UCB(left)
            ucb_right = self.compute_UCB(right)

            if ucb_left > ucb_right:
                node_key = left
            elif ucb_left < ucb_right:
                node_key = right
            else:
                random_val = np.random.uniform()
                if random_val > 0.5:
                    node_key = left
                else:
                    node_key = right

        return temp_node, path

    def expand_node(self, node_key):
        random_val = np.random.uniform()
        # Pick left unexplored child
        if random_val > 0.5:
            left = self.tree.get_left_child(node_key)
            return left

        else:
            right = self.tree.get_right_child(node_key)
            return right

    # Do a rollout to some leaf node at the bottom of the tree

    def rollout(self, node_key):
        temp_node = node_key
        while(node_key in self.tree.data):
            temp_node = node_key
            node_key = self.expand_node(node_key)

        return self.tree.data[temp_node].t / self.tree.data[temp_node].n

    # Backup the values from the unexplored node to the root (reverse tree policy path)

    def backup(self, node_key, path, reward):
        path.append(node_key)
        backup_path = path[::-1]

        for backup_key in backup_path:
            self.tree.data[backup_key].t += reward
            self.tree.data[backup_key].n += 1

    def perform_iters(self):
        for _ in range(self.nr_iters):
            node_key, path = self.select_node()
            expand_node_key = self.expand_node(node_key)
            if expand_node_key in self.tree.data and self.tree.data[expand_node_key].n >= 0:
                rollout_reward = self.rollout(expand_node_key)
                self.backup(expand_node_key, path, rollout_reward)


if __name__ == "__main__":
    mcts = MCTS(2, 3, 50, 5, 100)
    mcts.perform_iters()
    mcts.tree.pp_tree(0, 0)
    mcts.tree.debug_tree()
