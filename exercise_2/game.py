from utils import plot_rewards, plot_TD_results
from cell import Cell, SpecialCell
from mc import mc_policy_eval, plot_mc_heatmaps
from sarsa import plot_sarsa_heatmap, perform_sarsa
from qlearning import perform_qlearning, plot_qlearning_heatmap

import numpy as np


def main():
    dims = 9
    nr_episodes = 1000
    alpha, epsilon = 0.5, 0.5
    epsilon_decay = 0.99
    alpha_decay = 0.99
    decay_step = 100
    gamma = 1

    # Construct the grid
    rows, cols = (dims, dims)
    grid = [[Cell(True) for _ in range(cols)] for j in range(rows)]
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

    print("Plotting MC policy evaluation results.")
    # Mc policy eval
    states1 = mc_policy_eval(nr_episodes, grid, True,
                             dims, is_first_visit=True)
    states2 = mc_policy_eval(nr_episodes, grid, True,
                             dims, is_first_visit=False)
    states3 = mc_policy_eval(nr_episodes, grid, False,
                             dims, is_first_visit=True)
    states4 = mc_policy_eval(nr_episodes, grid, False,
                             dims, is_first_visit=False)

    states = [(True, True, states1), (True, False, states2),
              (False, True, states3), (False, False, states4)]

    plot_mc_heatmaps(states, dims)

    alphas = [0.2, 0.8]
    epsilons = [0.2, 0.8]
    decay_steps = [25, 100]

    print("Saving SARSA and Q-learning heatmaps and reward curves to results.")

    for a in alphas:
        for e in epsilons:
            for d in decay_steps:
                avg_rewards = []
                avg_qrewards = []
                avg_qvals = []
                avg_qqvals = []
                for i in range(1):
                    print(i)
                    # Sarsa
                    qvals, rewards = perform_sarsa(a, gamma,  grid, dims,
                                                   nr_episodes, e, epsilon_decay, alpha_decay, d)
                    # plot_sarsa_heatmap(qvals, dims)

                    # # Qlearning
                    qqvals, qrewards = perform_qlearning(
                        a, gamma,  grid, dims, nr_episodes, e, epsilon_decay, alpha_decay, d)
                    # plot_qlearning_heatmap(qqvals, dims)

                    avg_rewards.append(rewards)
                    avg_qrewards.append(qrewards)
                    avg_qvals.append(qvals)
                    avg_qqvals.append(qqvals)

                avg_rewards = np.average(avg_rewards, axis=0)
                avg_qrewards = np.average(avg_qrewards, axis=0)
                avg_qvals = np.average(avg_qvals, axis=0)
                avg_qqvals = np.average(avg_qqvals, axis=0)

                td_rewards = [avg_rewards, avg_qrewards]
                td_qvals = [avg_qvals, avg_qqvals]
                td_names = ["SARSA", "Q-learning"]
                plot_TD_results(td_qvals, td_rewards, td_names, dims,
                                nr_episodes, a, e, d)


if __name__ == "__main__":
    main()
