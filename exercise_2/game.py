from utils import plot_rewards, plot_TD_results
from cell import Cell, SpecialCell
from mc import mc_policy_eval, plot_mc_heatmaps
from sarsa import plot_sarsa_heatmap, perform_sarsa
from qlearning import perform_qlearning, plot_qlearning_heatmap


def main():
    dims = 9
    nr_episodes = 10000
    alpha, epsilon = 0.1, 0.7
    epsilon_decay = 0.95
    alpha_decay = 0.95
    decay_step = 200
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

    # # Mc policy eval
    # states1 = mc_policy_eval(nr_episodes, grid, True,
    #                          dims, is_first_visit=True)
    # states2 = mc_policy_eval(nr_episodes, grid, True,
    #                          dims, is_first_visit=False)
    # states3 = mc_policy_eval(nr_episodes, grid, False,
    #                          dims, is_first_visit=True)
    # states4 = mc_policy_eval(nr_episodes, grid, False,
    #                          dims, is_first_visit=False)

    # states = [(True, True, states1), (True, False, states2),
    #           (False, True, states3), (False, False, states4)]

    # plot_mc_heatmaps(states, dims)

    # Sarsa
    qvals, rewards = perform_sarsa(alpha, gamma,  grid, dims,
                                   nr_episodes, epsilon, epsilon_decay, alpha_decay, decay_step)
    # plot_sarsa_heatmap(qvals, dims)

    # # Qlearning
    qqvals, qrewards = perform_qlearning(
        alpha, gamma,  grid, dims, nr_episodes, epsilon, epsilon_decay, alpha_decay, decay_step)
    # plot_qlearning_heatmap(qqvals, dims)

    td_rewards = [rewards, qrewards]
    td_qvals = [qvals, qqvals]
    td_names = ["SARSA", "Q-learning"]
    # plot_rewards(rewards, qrewards, nr_episodes)
    plot_TD_results(td_qvals, td_rewards, td_names, dims,
                    nr_episodes, alpha, epsilon, alpha_decay, epsilon_decay, decay_step)
    # print(qrewards)


if __name__ == "__main__":
    main()
