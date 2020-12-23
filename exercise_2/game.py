import numpy as np
from numpy.lib.shape_base import _replace_zero_by_x_arrays
from cell import Cell, SpecialCell
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    dims = 9
    nr_episodes = 1000

    # Construct the grid
    rows, cols = (dims, dims)
    grid = [[Cell(True) for i in range(cols)] for j in range(rows)]
    unreachable = [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 7),
                   (4, 7), (5, 7), (6, 7), (8, 2), (8, 3), (8, 4), (8, 5)]

    # Because we are working with zero-based indexing.
    unreachable = [(i-1, j-1) for i, j in unreachable]

    for unreachable_state in unreachable:
        row, col = unreachable_state
        grid[row][col].is_reachable = False

    # Snake pit
    grid[6][5] = SpecialCell(True, -50)

    # Treasure
    grid[8][8] = SpecialCell(True, 50)

    states = mc_policy_eval(nr_episodes, grid, True,
                            dims, (7, 7), is_first_visit=True)

    reward_grid = np.resize(list(states.values()), (dims, dims))
    ax = sns.heatmap(reward_grid, annot=True, fmt=".2f")
    plt.xticks(np.arange(1, 10))
    ax.set_xticks(np.arange(0.5, 9))
    xlabels = [int(x + 0.5) for x in ax.get_xticks()]
    ylabels = [int(y + 0.5) for y in ax.get_yticks()]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    plt.show()


def generate_episode(starting_state, grid):
    episode = []
    temp_state = starting_state
    cur_state = starting_state
    row, col = temp_state
    while not isinstance(grid[row][col], SpecialCell):
        action_choice = np.random.uniform()
        #  Left
        if action_choice <= 0.25:
            temp_state = (row, col-1)
            episode.append("l")
        # Right
        elif action_choice <= 0.5:
            temp_state = (row, col+1)
            episode.append("r")
        # Up
        elif action_choice <= 0.75:
            temp_state = (row - 1, col)
            episode.append("u")
        # Down
        elif action_choice <= 1:
            temp_state = (row + 1, col)
            episode.append("d")

        temp_row, temp_col = temp_state
        if is_valid(temp_row, temp_col, grid):
            cur_state = temp_state
            row, col = cur_state
        else:
            episode.pop()
            episode.append("x")

    return episode, cur_state


def is_valid(temp_row, temp_col, grid):
    dim = len(grid)
    return temp_row >= 0 and temp_col >= 0 and temp_row <= (dim - 1) and temp_col <= (dim - 1) and grid[temp_row][temp_col].is_reachable


def set_unvisited(states, grid):
    for state in states:
        row, col = state
        grid[row][col].is_visited = False


def mc_policy_eval(nr_episodes, grid, is_random, dims, starting_state=(0, 0), is_first_visit=False):
    episode_counts = {(i, j): 0 for i in range(dims) for j in range(dims)}
    states_values = {(i, j): 0 for i in range(dims) for j in range(dims)}
    states_indices = np.arange(dims*dims)
    states = list(states_values.keys())

    for _ in range(nr_episodes):
        # Pick random starting state
        if is_random:
            while True:
                starting_state_index = np.random.choice(states_indices)
                starting_state = states[starting_state_index]
                start_row, start_col = starting_state
                start_cell = grid[start_row][start_col]
                if start_cell.is_reachable and not isinstance(start_cell, SpecialCell):
                    break

        episode, cur_state = generate_episode(starting_state, grid)

        # Reverse the episode and start from the back
        # Set visited to False when starting new episode.
        row, col = cur_state
        episode = episode[::-1]
        accum_reward = 0
        set_unvisited(states, grid)

        for dir in episode:
            prev_state = cur_state
            accum_reward += grid[row][col].reward
            if dir == "r":
                # Then we need to go left on the grid, because we accumulate in a tailwise
                cur_state = (row, col - 1)
            elif dir == "l":
                cur_state = (row, col + 1)
            elif dir == "u":
                cur_state = (row + 1, col)
            elif dir == "d":
                cur_state = (row - 1, col)

            row, col = cur_state

            if grid[row][col].is_reachable:
                if not is_first_visit:
                    states_values[cur_state] += accum_reward
                    episode_counts[cur_state] += 1
                elif not grid[row][col].is_visited:
                    states_values[cur_state] += accum_reward
                    episode_counts[cur_state] += 1
                    grid[row][col].is_visited = True

    # print(episode_counts)
    states_values = {k: v/episode_counts[k] if episode_counts[k]
                     != 0 else v for k, v in states_values.items()}
    return states_values


if __name__ == "__main__":
    main()
