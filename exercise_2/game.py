from utils import plot_rewards
from cell import Cell, SpecialCell
from mc import mc_policy_eval, plot_mc_heatmap
from sarsa import plot_sarsa_heatmap, perform_sarsa
from qlearning import perform_qlearning, plot_qlearning_heatmap


def main():
    dims = 9
    nr_episodes = 10000
    alpha = 0.1
    epsilon = 0.3
    epsilon_decay = 0.9995
    alpha_decay = 1  # 0.9995

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

    # Mc policy eval
    # states = mc_policy_eval(nr_episodes, grid, True,
    #                         dims, (7, 7), is_first_visit=True)
    # plot_mc_heatmap(states, dims)

    # Sarsa
    qvals, rewards = perform_sarsa(alpha, grid, dims,
                                   nr_episodes, epsilon, epsilon_decay, alpha_decay)
    plot_sarsa_heatmap(qvals, dims)

    # Qlearning
    qqvals, qrewards = perform_qlearning(
        alpha, grid, dims, nr_episodes, epsilon, epsilon_decay, alpha_decay)
    plot_qlearning_heatmap(qqvals, dims)

    plot_rewards(rewards, qrewards, nr_episodes)


if __name__ == "__main__":
    main()
