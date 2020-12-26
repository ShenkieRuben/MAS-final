import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import pick_random_state, is_valid, get_epsilon_greedy_action
from cell import SpecialCell


def plot_sarsa_heatmap(qvals, dims):
    policy_indices = np.argmax(qvals, axis=1)
    max_q = np.max(qvals, axis=1)
    actions = ['l', 'r', 'u', 'd']
    actions_symbols = {'l': '<', 'r': '>', 'u': '^', 'd': 'v'}
    policy_steps = [actions[i] for i in policy_indices]

    labels = [f"{round(val, 3)}\n {actions_symbols[act]}" if val != 0 else "" for val,
              act in zip(max_q, policy_steps)]
    ax = sns.heatmap(max_q.reshape(dims, dims), annot=np.reshape(
        labels, (dims, dims)), fmt="", cmap='RdYlGn')

    plt.xticks(np.arange(1, 10))
    ax.set_xticks(np.arange(0.5, 9))
    xlabels = [int(x + 0.5) for x in ax.get_xticks()]
    ylabels = [int(y + 0.5) for y in ax.get_yticks()]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    plt.show()


def perform_sarsa(alpha, grid, dims, nr_episodes, epsilon, epsilon_decay, alpha_decay):
    # Dict to convert states to indices.
    states = {(i, j): i * dims + j
              for i in range(dims) for j in range(dims)}

    state_indices = list(states.values())
    state_vals = list(states.keys())

    # 0 = left
    # 1 = right
    # 2 = up
    # 3 = down

    actions = ['l', 'r', 'u', 'd']
    accum_reward = []

    # Perhaps change this to random qvals.
    # qvals = np.random.random((dims * dims, len(actions)))
    qvals = np.zeros((dims * dims, len(actions)))
    for i in range(nr_episodes):
        starting_state = pick_random_state(state_indices, state_vals, grid)
        # starting_state = (0, 0)
        # Initialize values and s and a.
        new_state = starting_state
        cur_state = starting_state
        action_index = get_epsilon_greedy_action(
            qvals, epsilon, cur_state, states, actions)
        action = actions[action_index]
        row, col = cur_state

        epsilon *= epsilon_decay
        alpha *= alpha_decay
        total_reward = 0

        while not isinstance(grid[row][col], SpecialCell):
            if action == "l":
                new_state = (row, col-1)
            # Right
            elif action == "r":
                new_state = (row, col+1)
            # Up
            elif action == "u":
                new_state = (row - 1, col)
            # Down
            elif action == "d":
                new_state = (row + 1, col)

            new_row, new_col = new_state

            if is_valid(new_row, new_col, grid):
                reward = grid[new_row][new_col].reward
            else:
                # Bumped into a wall or gone outside grid
                new_state = cur_state
                reward = -1

            total_reward += reward

            new_action_index = get_epsilon_greedy_action(
                qvals, epsilon, new_state, states, actions)
            new_action = actions[new_action_index]

            qsa = qvals[states[cur_state], action_index]
            new_qsa = qvals[states[new_state], new_action_index]

            qvals[states[cur_state],
                  action_index] += alpha * (reward + new_qsa - qsa)

            cur_state = new_state
            row, col = cur_state
            action_index = new_action_index
            action = new_action

        accum_reward.append(total_reward)

    return qvals, accum_reward
