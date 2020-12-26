import numpy as np
from cell import SpecialCell

import matplotlib.pyplot as plt


def is_valid(temp_row, temp_col, grid):
    dim = len(grid)
    return temp_row >= 0 and temp_col >= 0 and temp_row <= (dim - 1) and temp_col <= (dim - 1) and grid[temp_row][temp_col].is_reachable


def set_unvisited(states, grid):
    for state in states:
        row, col = state
        grid[row][col].is_visited = False


# Pick random state that is not a wall or absorbing state.
def pick_random_state(states_indices, state_tuples, grid):
    while True:
        starting_state = state_tuples[np.random.choice(states_indices)]
        row, col = starting_state
        start_cell = grid[row][col]
        if start_cell.is_reachable and not isinstance(start_cell, SpecialCell):
            break

    return starting_state


def plot_rewards(rewards, qrewards, nr_episodes):
    epis = np.arange(nr_episodes)

    plt.plot(epis, rewards)
    plt.plot(epis, qrewards)
    plt.show()


def get_epsilon_greedy_action(qvals, epsilon, cur_state, states, actions):
    prob = np.random.uniform()

    # Pick greedy action index
    if prob <= (1 - epsilon):
        index = np.argmax(qvals[states[cur_state]])
    else:
        index = np.random.randint(0, len(actions))

    return index
