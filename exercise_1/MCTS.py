from random import random
from binary_tree import Tree
import math
import numpy as np


class MCTS():
    def __init__(self, c, d, nr_iters, nr_rollouts):
        self.c = c
        self.d = d
        self.nr_iters = nr_iters
        self.nr_rollouts = nr_rollouts

        # Maintain current root and actual path
        self.cur_root = 0
        self.informed_path = [0]

        num_leaves = 2 ** self.d

        self.tree = Tree()
        for _ in range(num_leaves):
            self.tree.insert()

        self.tree.assign_vals_to_leafs(num_leaves)

    # def compute_xbar(self, node_key):

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
        node_key = self.cur_root
        path = []

        # The root, special case
        if self.tree.data[node_key].n == 0:
            path.append(node_key)
            node_key = self.get_random_child(node_key)
            path.append(node_key)
            return node_key, path

        # Larger than 0 to prevent computing UCB values on leaves.

        while node_key in self.tree.data and self.tree.data[node_key].n > 0:
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

        path.append(node_key)

        return node_key, path

    def get_random_child(self, node_key):
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
            node_key = self.get_random_child(node_key)

        return self.tree.data[temp_node].t / self.tree.data[temp_node].n

    # Backup the values from the unexplored node to the root (reverse tree policy path)

    def backup(self, path, reward):
        backup_path = path[::-1]
        for backup_key in backup_path:
            self.tree.data[backup_key].t += reward
            self.tree.data[backup_key].n += 1

    def perform_iters(self):
        while self.cur_root in self.tree.data:
            for _ in range(self.nr_iters):
                node_key, path = self.select_node()
                # Check if node key is leaf
                if self.tree.data[node_key].n == 0:
                    # Perform a number of rollouts from that node.
                    for _ in range(self.nr_rollouts):
                        rollout_reward = self.rollout(node_key)
                        self.backup(path, rollout_reward)

            decision_node = self.make_informed_decision()
            if decision_node == -1:
                break
            else:
                self.cur_root = decision_node
                self.informed_path.append(self.cur_root)

    def make_informed_decision(self):
        node_key = self.cur_root

        left = self.tree.get_left_child(node_key)
        right = self.tree.get_right_child(node_key)

        if left in self.tree.data and right in self.tree.data:
            left_xbar = self.tree.data[left].t / self.tree.data[left].n
            right_xbar = self.tree.data[right].t / self.tree.data[right].n

            if left_xbar > right_xbar:
                node_key = left
            elif right_xbar > left_xbar:
                node_key = right
            else:
                node_key = self.get_random_child(node_key)
        else:
            # Non-existing
            return -1

        return node_key


if __name__ == "__main__":

    # mcts.tree.pp_tree(0, 0)
    # mcts.tree.debug_tree()

    mcts.tree.print_max()
    # print(mcts.informed_path)
    # found = mcts.tree.data[mcts.informed_path[-1]]
    # print(found.t/found.n)
    pass
