from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

from MCTS import MCTS


def set_ranges(depths, nr_its):
    return np.arange(1, depths + 1), np.arange(nr_its)


def plot_regret(cvals, depths, nr_its):
    ds, its = set_ranges(depths, nr_its)
    ds_regret = {}
    perf = {}

    for d in ds:
        ds_regret[d] = []
        perf[d] = []

    for it in its:
        print(it)
        for d in ds:
            regrets = []
            for c in cvals:
                mcts = MCTS(c, d, 50, 5)
                mcts.perform_iters()
                max_val = mcts.tree.get_max()
                found_leave = mcts.tree.data[mcts.informed_path[-1]]
                found_val = found_leave.t / found_leave.n
                diff = max_val - found_val
                regrets.append(diff)
            ds_regret[d].append(regrets)

    for d in ds:
        regret_matrix = np.array(ds_regret[d])
        avgs = np.average(regret_matrix, axis=0)
        stds = np.std(regret_matrix, axis=0)
        perf[d] = (avgs, stds)

    print(perf)
    total_avgs = []
    total_stds = []

    for i in range(len(cs)):
        c_lines_avgs = []
        c_lines_stds = []
        for key, val in perf.items():
            cval_avg = val[0][i]
            cval_std = val[1][i]
            c_lines_avgs.append(cval_avg)
            c_lines_stds.append(cval_std)
        total_avgs.append(c_lines_avgs)
        total_stds.append(c_lines_stds)

    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 2)

    for line, ax in enumerate(fig.axes):
        col = next(palette)
        ax.errorbar(ds, total_avgs[line],
                    total_stds[line], fmt="o-", capsize=5,  label=f"c={cs[line]}", color=col, lw=2, alpha=0.5)
        # ax.plot(ds, total_avgs[line],
        #         label=f"c={cs[line]}", color=col, alpha=0.5)
        for d in range(len(ds)):
            if d % 2 == 0:
                ax.annotate(f"{round(total_avgs[line][d],2)}",
                            xy=(ds[d], total_avgs[line][d]))

        ax.set_xticks(ds)
        ax.legend()

    fig.suptitle("Average total regret for varying depths and c values.")
    fig.text(0.5, 0.04, 'Depth (d)', va='center', ha='center')
    fig.text(0.04, 0.5, 'Average regret', va='center', ha='center',
             rotation='vertical')

    plt.show()

    print(total_avgs)
    print(total_stds)


def plot_perf(depths, nr_its):

    ds, its = set_ranges(depths, nr_its)

    ds_time = {}
    perf = {}

    for d in ds:
        ds_time[d] = []
        perf[d] = ()

    for it in its:
        print(it)
        for d in ds:
            start = timer()
            mcts = MCTS(2, d, 50, 5)
            mcts.perform_iters()
            end = timer()
            ds_time[d].append(end - start)

    for key, val in ds_time.items():
        time_results = val
        perf[key] = (np.mean(time_results), np.std(time_results))

    res = list(perf.values())
    res = list(zip(*res))
    means = res[0]
    stds = res[1]
    plt.errorbar(list(perf.keys()), means, stds, fmt="-", capsize=5)
    plt.title("Performance of MCTS")
    plt.xlabel("Depth (d)")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(0, len(perf)+1, 1))
    plt.show()


if __name__ == "__main__":
    cs = [0.1, 1, 5, 10, 50, 100]
    # cs = [0.1, 100]

    # plot_perf(10, 100)
    plot_regret(cs, 16, 100)
