import numpy as np
import pandas as pd
from cell import SpecialCell

import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil


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


def plot_TD_results(td_qvals, td_rewards, td_names, dims, nr_episodes, alpha, epsilon, decay_step):
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
    # ax3 = fig.add_subplot(1, 3, 3)
    # axs = [ax1, ax2, ax3]

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(19.2, 10.8)

    actions = ['l', 'r', 'u', 'd']
    actions_symbols = {'l': '<', 'r': '>', 'u': '^', 'd': 'v'}

    # Reshape the two arrays into one matrix and check the minimum/maximum values in the heatmaps.
    flattened_q = np.asarray(td_qvals).reshape((dims*dims*2, 4))
    chosen_q = np.max(flattened_q, axis=1)
    chosen_q = chosen_q[chosen_q != 0]
    min_qs = floor(np.min(chosen_q))
    max_qs = floor(np.max(chosen_q))

    cbar_ticks = np.floor(np.linspace(min_qs, max_qs, 5))
    cbar_ticks[0] = floor(cbar_ticks[0])
    cbar_ticks[-1] = ceil(cbar_ticks[-1])
    cbar_ticks = np.floor(cbar_ticks)

    # Plot SARSA and Q-learning heatmaps.
    for i in range(2):
        qvals = td_qvals[i]
        max_q = np.max(qvals, axis=1)
        policy_indices = np.argmax(qvals, axis=1)
        policy_steps = [actions[i] for i in policy_indices]

        # Prevent printing qvals of walls and absorbing states
        labels = [f"{round(val, 2)} \n {actions_symbols[act]}" for val, act in zip(
            max_q, policy_steps)]
        max_q = max_q.reshape(dims, dims)
        ax = sns.heatmap(max_q, annot=np.reshape(labels, (dims, dims)), mask=max_q == 0, fmt="", cmap="Spectral",
                         ax=axs[i], vmin=min_qs, vmax=max_qs, cbar=i == 1, cbar_kws=dict(ticks=cbar_ticks))
        ax.set_facecolor("xkcd:light grey")

        axs[i].set_title(td_names[i])

        # Set the positions of the xticks to be in the middle.
        axs[i].set_xticks(np.arange(0.5, dims))
        xtickpos = [int(x + 0.5) for x in axs[i].get_xticks()]
        ytickpos = [int(y + 0.5) for y in axs[i].get_yticks()]
        axs[i].set_xticklabels(xtickpos)
        axs[i].set_yticklabels(ytickpos)

    # Plot rewards
    epis = np.arange(nr_episodes)
    axs[2].plot(epis, pd.Series(td_rewards[0]).rolling(
        5).mean(), label="SARSA")
    axs[2].plot(epis, pd.Series(td_rewards[1]).rolling(
        5).mean(), label="Q-learning")
    # axs[2].plot(epis, td_rewards[0], label="SARSA")
    # axs[2].plot(epis, td_rewards[1], label="Q-learning")
    axs[2].set_title("Number of episodes (x) \n Accumulated rewards (y)")
    axs[2].legend()

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.suptitle(
        f"SARSA vs Q-learning with α={alpha}, ε={epsilon} and decay_step={decay_step}")
    plt.subplots_adjust(top=0.887,
                        bottom=0.051,
                        left=0.025,
                        right=0.985,
                        hspace=0.2,
                        wspace=0.109)
    fig.savefig(os.path.join(
        "results", f"td_alph={alpha}_eps={epsilon}_ds={decay_step}.png"), bbox_inches='tight', dpi=200)
    # plt.show()


def plot_rewards(rewards, qrewards, nr_episodes):
    epis = np.arange(nr_episodes)

    plt.plot(epis, rewards, label="SARSA")
    plt.plot(epis, qrewards, label="Q-learning")

    plt.xlabel("Number of episodes")
    plt.ylabel("Accumulated reward")
    plt.legend()
    plt.show()


def get_epsilon_greedy_action(qvals, epsilon, cur_state, states, actions):
    prob = np.random.uniform()

    # Pick greedy action index
    if prob <= (1 - epsilon):
        index = np.argmax(qvals[states[cur_state]])
    else:
        index = np.random.randint(0, len(actions))

    return index
