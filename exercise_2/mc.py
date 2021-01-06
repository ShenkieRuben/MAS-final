import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cell import SpecialCell
from utils import *


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


def mc_policy_eval(nr_episodes, grid, is_random, dims, starting_state=(0, 0), is_first_visit=False):
    episode_counts = {(i, j): 0 for i in range(dims) for j in range(dims)}
    states_values = {(i, j): 0 for i in range(dims) for j in range(dims)}
    states_indices = np.arange(dims*dims)
    states = list(states_values.keys())

    for _ in range(nr_episodes):
        # Pick random starting state
        if is_random:
            starting_state = pick_random_state(states_indices, states, grid)

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


def plot_mc_heatmaps(states, dims):

    fig, axs = plt.subplots(2, 2)

    for i, ax in enumerate(fig.axes):
        random_starting_state = states[i][0]
        first_visit = states[i][1]
        mc_res = states[i][2]
        reward_grid = np.resize(list(mc_res.values()), (dims, dims))
        sns.heatmap(reward_grid, annot=True, fmt=".2f", ax=ax, cmap="Spectral")
        ax.set_xticks(np.arange(0.5, 9))
        xlabels = [int(x + 0.5) for x in ax.get_xticks()]
        ylabels = [int(y + 0.5) for y in ax.get_yticks()]
        ax.set_title(
            f"Random starting state: {random_starting_state} and first visit: {first_visit}")
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

    plt.show()
